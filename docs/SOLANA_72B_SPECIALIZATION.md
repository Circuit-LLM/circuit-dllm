# Specializing the 72B on Solana — a swap runbook

**Goal:** produce a Solana-specialized model that is a **fine-tune of the SAME base we already serve
(`Qwen/Qwen2.5-72B-Instruct`)**, AWQ-quantize it, and hot-swap it behind the floating coordinator —
**without disturbing predictive drafting, tree drafting, the prefix cache, the control plane, or HA.**

**Status:** plan / runbook. Companion to the data plan (the on-chain corpus + the swarm's labeled
outcomes) and the AWQ serving docs (`docs/AWQ_PER_NODE.md`).

---

## 0. The one rule that keeps this a drop-in

Stay inside the **Qwen2.5-72B architecture**. Same config (80 decoder layers, same dims), **same
tokenizer/vocab**. If that holds, the swap is just "new weights + re-quant + refresh draft," and
everything else we built keeps working. The moment you change model family, the tokenizer changes,
which kills the draft model and forces a re-slice — a different, much larger project. This runbook
assumes we hold the rule.

```
 corpus ─▶ [A] continued pretrain ─▶ [B] SFT (QLoRA) ─▶ [C] RL on outcomes ─▶ final bf16 72B
                  (H100 cluster)        (1–2 GPUs)         (1–2 GPUs)
 final bf16 72B ─▶ [D] AWQ quantize ─▶ [E] publish/slice shards ─▶ [F] refresh draft
 [G] validate (quality + slice integrity + acceptance rate) ─▶ [H] parallel bring-up → A/B → flip → keep old warm
```

---

## 1. Invariants (do not change these)

| Invariant | Why |
|---|---|
| Base config = `Qwen2.5-72B-Instruct` (80 layers) | Existing layouts (`[0,40)+[40,80)`) + `slice-awq.py` work unchanged |
| Tokenizer / vocab unchanged | The 0.5B draft + all tooling stay valid; speculative decoding stays lossless |
| Serving stays **AWQ** (Marlin) | The mesh loads AWQ slices; we re-quant, never switch quant scheme mid-swap |
| Draft stays in the Qwen2.5 family | `CIRCUIT_DRAFT` keeps pointing at a Qwen draft; only the weights refresh |

---

## 2. Data (inputs to each stage)

Pull from the data plan; in short:
- **Continued-pretrain corpus (A):** Solana docs + program IDLs/source, serialized on-chain event
  sequences (from the firehose archive + RPC backfill), market history (indexer→Postgres candles),
  security forensics, curated social/news. Target a few **billion tokens**, quality-filtered.
- **SFT set (B):** instruction + **tool-use transcripts** (`question → which data-api endpoints →
  reason over JSON → answer/act`), structured/JSON outputs, scam-classification Q&A. Bootstrap with
  the live 72B (self-distillation) + hand-curated golden sets. Tens of thousands → low millions.
- **RL set (C):** the swarm's `signal → outcome` pairs + reasoning traces
  (`conversation_archive.jsonl`, `trading_notes`, `solana_trades.json`) as a **real reward signal**.

---

## 3. Stage A — continued pretraining (the cluster job)

The only stage that needs a real cluster; everything after fits on 1–2 GPUs.

- **Hardware:** RunPod **8–32× H100 80GB** (Instant Cluster / reserved multi-node, fast interconnect).
  16× H100 is a sensible middle; 8× works with more offload + time.
- **Stack:** HF `transformers` + **DeepSpeed ZeRO-3** (or FSDP) with optimizer/param offload;
  flash-attention; gradient checkpointing; bf16. (`scripts/sync-runpod.sh` / `prov-pod.py` already
  stand pods up.)
- **Regime:** *continued* pretraining, not from scratch — **low LR** (~1e-5 → 1e-6 cosine), 1–3
  passes over the corpus, **replay a slice of general data (~5–15%)** to avoid catastrophic
  forgetting (we still want a smart general model, just Solana-fluent).
- **Checkpoint** to object storage every N steps (a 72B run will get pre-empted; resume-ability is
  mandatory).
- **Output:** a bf16 `Qwen2.5-72B-Instruct-solana` checkpoint.
- **Cost/time:** days–weeks, low-thousands → low-tens-of-thousands $. This is the expensive line item;
  do it once per major data refresh, not continuously.

> Cheaper alternative if budget is tight: skip full continued-pretrain and bake Solana knowledge via
> a **high-rank LoRA** over a large pretrain-style corpus on 1–2 GPUs. Less deep than a full pass, but
> 10–50× cheaper and a fine first iteration.

---

## 4. Stage B — SFT with QLoRA (1–2 GPUs)

- **Hardware:** one **H200 141GB** (or 2× H100). QLoRA = 4-bit frozen base + trainable adapters, so
  72B fits comfortably.
- **Stack:** `peft` QLoRA (NF4), `trl` SFTTrainer, packed sequences, LR ~1e-4 on adapters, 1–3 epochs.
- **Train on the tool-use transcripts hardest** — this is what makes it an *agent* that calls our
  data-api correctly, not a chatbot. Enforce the function-call/JSON schema in the targets.
- **Merge** the adapter into bf16 for the next stage (or keep stacking adapters).
- **Cost/time:** hours–1 day, low hundreds $.

---

## 5. Stage C — RL on real outcomes (1–2 GPUs)

The part competitors can't copy — we have a live reward signal.

- **Method:** **DPO or GRPO** over the swarm's outcomes (preference = the decision that actually made
  money vs the one that didn't / the rugged one). DPO is simplest and stable on QLoRA.
- **Reward shaping:** weight **downside heavily** — avoiding a rug must score far above catching a
  pump. We're training discipline and risk-filtering, not pump-chasing.
- **Keep a KL leash** to the SFT model so RL sharpens judgment without wrecking general ability.
- **Output:** the final bf16 `…-solana` checkpoint to quantize.
- **Cadence:** this is the **flywheel** — re-run as new labeled outcomes accrue (weekly/monthly).

---

## 6. Stage D — AWQ quantization

The fine-tune is bf16; the mesh serves AWQ. Quantize the **full** new 72B before slicing.

- Run **AWQ calibration** (autoawq) on a representative calibration set (mix of general + Solana
  prompts) → a full AWQ/Marlin checkpoint, same format as today's `Qwen2.5-72B-Instruct-AWQ`.
- ⚠️ **Known gotcha:** the head-only / 0-layer AWQ slice must drop `quantization_config` or
  transformers' AWQ replace path throws `has_been_replaced UnboundLocalError` — already handled in
  `scripts/slice-awq.py`; make sure the freshly-quantized checkpoint carries the same config shape the
  slicer expects.
- **Quant QA:** perplexity delta bf16 → AWQ on a held-out set (expect a small, bounded drop). If the
  drop is large, recalibrate.

---

## 7. Stage E — publish + slice the shards

- `python3 scripts/publish-awq-shards.py <full_awq_dir> /root/shards --layout 0:40,40:80 --repo
  circuitllmdev/qwen25-72b-solana-awq-shards --upload` → sha256-verified shards on HF.
- Mesh nodes either **download published shards** (`CIRCUIT_AWQ_SHARDS=circuitllmdev/qwen25-72b-solana-awq-shards`,
  integrity-verified) or **slice locally** with `scripts/slice-awq.py <ckpt> <out> <start> <end>`.
- Validate the round-trip with `scripts/validate-awq-download.sh` (publish → download → integrity →
  submodel-loaded → ready) against the new repo before any node serves it.

---

## 8. Stage F — refresh the draft (for speed, not correctness)

Speculative decoding is **lossless by construction**: even with today's 0.5B draft, the new 72B's
output is exactly correct. What can drop is the **acceptance rate** (the fine-tuned target drifts from
what the old draft predicts), shrinking the ~13–18 tok/s win. To keep the speed:

- **Fine-tune the draft** (`Qwen/Qwen2.5-0.5B-Instruct`, optionally 1.5B) on the same SFT data, or
  **distill** from the new 72B (train the draft to mimic the new target's next-token distribution).
- Point `CIRCUIT_DRAFT` at the refreshed draft.
- Re-measure the **tree-drafting** acceptance and re-tune **K** (currently 12) — the optimum may shift
  a little. The tree-draft *code* is unchanged; this is a config sweep.

---

## 9. Stage G — validation (pre-swap)

Byte-identical validation does **not** apply here (weights changed on purpose). Instead, gate on four
checks:

1. **Slice integrity** — `validate-awq-download.sh` green on the new shards repo.
2. **Quality evals** — held-out Solana QA accuracy, **tool-use exact-match** (does it call the right
   data-api endpoints with valid args), scam-classification F1, and a general-ability regression set
   (it must not get dumber at normal chat).
3. **Throughput / acceptance** — `scripts/staging-throughput.py` (or a 72B variant) to confirm the
   refreshed draft holds the tok/s and concurrency we had.
4. **Canary diff** — a fixed prompt battery, new vs current model, eyeballed for regressions.

Ship only when all four pass.

---

## 10. Stage H — hot-swap + rollback

Reuse the proven floating-coordinator cutover (we already did a byte-identical swap of orchestrators +
holders with no downtime):

1. Bring up a **parallel** stack — new orchestrators + holders pointing at
   `CIRCUIT_AWQ_SHARDS=…solana…` + the refreshed `CIRCUIT_DRAFT` — registered to the **same control
   plane** but not yet advertised as the served model.
2. **A/B** through the canary battery against live traffic mirror.
3. **Flip** the control plane to advertise the new model id / route to the new holders.
4. Keep the **old stack warm and resumable** (EXITED-not-deleted) for **instant rollback** — exactly
   how the last cutover was kept reversible.

The coordinator, control plane (incl. on-chain control plane), HA failover, and prefix cache are all
**weights-agnostic** — none of them change for this swap.

---

## 11. What carries over vs what you redo

| Untouched (weights-agnostic) | Redo for the swap |
|---|---|
| Floating coordinator + mesh topology / layer split | **Re-quantize to AWQ** (Stage D) |
| Prefix cache (TTFT) | **Refresh the draft** (Stage F) — for speed |
| Tree-drafting *code* | **Re-tune K** + re-measure acceptance |
| Control plane / on-chain control plane / HA | Publish new shards repo (Stage E) |
| The `circuit agent` / gateway / x402 layer | A/B + flip (Stage H) |

---

## 12. Cost, time, cadence

| Stage | Hardware | Time | Rough cost | Cadence |
|---|---|---|---|---|
| A · continued pretrain | 8–32× H100 cluster | days–weeks | low-thousands → low-tens-of-thousands $ | per major data refresh |
| B · SFT (QLoRA) | 1× H200 / 2× H100 | hours–1 day | hundreds $ | frequent |
| C · RL on outcomes | 1× H200 / 2× H100 | hours–1 day | hundreds $ | the flywheel (weekly/monthly) |
| D–E · AWQ + publish | 1× H100/H200 | hours | tens $ | every swap |
| F · draft refresh | 1× consumer/H100 | hours | tens $ | every swap |
| G–H · validate + swap | the mesh | hours | ~free | every swap |

**Pragmatic first iteration:** skip the cluster (Stage A → high-rank LoRA on 1–2 GPUs), do B+C+D+E+F,
swap, measure. Prove the whole pipeline cheaply, then spend cluster money on the full continued-pretrain
once the data + evals justify it. And consider proving it on the **14B agent model first** — same
pipeline, a fraction of the cost and risk.

---

## 13. Risks & gotchas

- **Catastrophic forgetting** in Stage A → keep a general-data replay slice + a general-ability eval in
  the gate (Stage G #2). A Solana savant that can't reason isn't the goal.
- **AWQ quality cliff** → always check the perplexity delta; recalibrate if large.
- **Acceptance-rate regression** → the speed loss is real if you skip Stage F; never swap the target
  without refreshing the draft (or you'll quietly halve the tok/s).
- **Tokenizer drift** → if any stage touches the tokenizer/vocab (don't), every downstream invariant in
  §1 breaks. Treat the tokenizer as immutable.
- **One-way deploys** → never flip without the old stack warm (Stage H #4).
