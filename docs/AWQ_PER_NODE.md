# AWQ per node — pre-sliced sub-checkpoints (SPEED_ROADMAP §1.2 / mesh lever #2)

**Status:** ✅ VALIDATED + MEASURED ON 72B (2026-06-23). A real **1-hop** mesh — coordinator
keep-head slice `[0,48)` on an **L40S** (48GB) + stage `[48,80)` on an **L4** (24GB), 1.5B draft,
predictive drafting — served the 72B-AWQ at **13.66 tok/s warm single-stream** (acceptance 0.55 /
3.2 tok-per-round), ≈ the production 32B (13-14) and ~82% of the A100-AWQ solo ceiling (16.67),
and **≈4.9× the modest bnb 4-GPU mesh (2.77)**. Aggregate saturates ~14-15 tok/s (a 2-stage
pipeline overlaps ~2 requests). The first (cold) request reads ~2.8 tok/s — pure Marlin-warmup +
first stage-connection; warm is the real number. Bottleneck = the slowest card's memory bandwidth
(the L4 stage); a balanced fat-card mesh would push single-stream higher. **The two levers
together (fewest-fattest 1 hop + AWQ-per-node Marlin) make the decentralized 72B mesh "fast
enough."** Both bench pods torn down; prod 32B untouched.

**Earlier component validation (Qwen2.5-7B-AWQ, A5000):** gptqmodel loads a pre-sliced
8-layer sub-checkpoint on cuda (Marlin selected, no error) **even with embed_tokens + lm_head
dropped** (the lean form — no fallback needed). The sliced layers' weights are **bytewise-identical**
to the full model's (all 320 tensors), and the stage forward matches the full model **within the
fp16 nondeterminism floor** (sub-vs-full Δ 7.0e-2 < full-vs-full Δ 9.4e-2). `tests/test_awq_subslice_gpu.py`.
Open question #1 (does gptqmodel load an n-layer sub-checkpoint?) is ANSWERED YES. Remaining:
the `stage_worker` sub-model load path + a full stitched-mesh greedy-equivalence run on the 72B.

## Why
The mesh runs **bnb** 4-bit today because **AWQ can't shard**: gptqmodel's Marlin kernel runs
`post_init` on *every* layer and requires each on `cuda`; a sharded load leaves un-owned layers
on `meta`/`cpu` → `Expected a cuda device, but got: meta`. So the only quant that shards is bnb
(quantize-at-load from fp16, un-owned layers never materialized — `load_model_shard_bnb`).

But bnb is **2.1× slower than AWQ/Marlin** (measured on one A100: bnb 7.93 vs AWQ 16.67 tok/s
single-stream). A bnb mesh, even with fewest-fattest hops, is bounded by bnb compute and can't
approach the AWQ ceiling. Making each node run **AWQ/Marlin on its slice** is the single biggest
per-node compute lever for a fast mesh.

## The idea — sidestep sharding, don't fight it
Instead of loading one 80-layer AWQ checkpoint with most layers pruned/meta (which Marlin
rejects), **pre-slice the AWQ checkpoint offline into one self-consistent sub-model per stage.**
Each sub-model is a *complete* small AWQ model: its `config.json` says `num_hidden_layers = n`,
it contains exactly that stage's `n` layers renumbered `0..n-1`, and its quantized tensors are
all present. A node loads it as a normal AWQ model → **every layer is on cuda → Marlin's
post_init is happy.** No meta, no pruning, no sharding. The coordinator stitches the stages over
the wire exactly as it does today; only the per-node *load* changes.

```
full:   model.layers.0..79  (+ embed_tokens, norm, lm_head)   one AWQ checkpoint
                │ slice [16,48)            │ slice [48,80)
stage A:  layers 0..31  ◄ global 16..47    stage B: layers 0..31 ◄ global 48..79
          num_hidden_layers=32                       num_hidden_layers=32
          (no lm_head; embed→Identity at runtime)     (coordinator does embed/norm/lm_head)
```

## What a stage sub-model contains
- **Decoder layers only**, renumbered: global `model.layers.{i}` → local `model.layers.{i-start}`
  for `i in [start,end)`. These carry the AWQ qweight/qzeros/scales — the only quantized tensors.
- **`model.norm`** — keep optionally (the final stage's output is normed on the *coordinator*,
  which holds `norm`+`lm_head`+embed; a pure stage doesn't need it). Default: drop.
- **`model.embed_tokens` / `lm_head`** — dropped. The stage runs on incoming hidden states, not
  token ids, and never projects to logits. (lm_head is ~2.4GB fp16 for the 72B vocab — dropping
  it is a real VRAM saving per node.) At runtime the loaded sub-model's `embed_tokens` is set to
  `Identity` so a stray id-path can't touch it; we feed hidden states straight into the layers.
- **`config.json`** — copied verbatim EXCEPT `num_hidden_layers = end-start` and
  `tie_word_embeddings` forced false (no lm_head to tie). `quantization_config` is preserved so
  the loader takes the AWQ/Marlin path.

## The slicer — `scripts/slice-awq.py`
Two parts:
1. **Pure remap core (stdlib only, unit-tested offline, `tests/test_slice_awq.py`):**
   - `remap_weight_map(weight_map, start, end, keep_norm=False)` — filters the safetensors
     index's `weight_map` (tensor-name → shard-file) to the kept tensors with layers renumbered.
   - `sub_config(config, start, end)` — the sub-model config dict.
   These are deterministic dict/string transforms — no torch, no safetensors.
2. **Pod-side I/O (`main`, lazy-imports `safetensors`):** reads the full checkpoint's
   `model.safetensors.index.json`, computes the remap, copies each kept tensor into the
   sub-model's shards (renamed), writes the sub-model `config.json` + a fresh index + the
   tokenizer files (harmless). Run once per stage on a pod that has the full checkpoint staged.

## Open questions — what GPU validation must confirm
1. **Does gptqmodel load an n-layer sub-checkpoint cleanly?** Expectation: yes — the dir is
   self-consistent (config + index + tensors all agree on `n` layers), so Marlin sees `n` cuda
   layers and post_init passes. RISK: gptqmodel may expect `lm_head`/`embed_tokens` present, or
   key off the *original* layer count somewhere. Fallback: keep a tiny real `lm_head` slice, or
   load as `AutoModel` (no CausalLM head) instead of `AutoModelForCausalLM`.
2. **Numerical correctness:** the stitched mesh output (stage A local 0..31 = global 16..47,
   etc.) must match the full-model greedy decode bitwise on CPU fp32 and within fp16 tolerance on
   GPU. Reuse the `test_quant_split` / stage-equivalence harness.
3. **Rotary / position handling:** each sub-model builds its own `rotary_emb`; positions are
   global (the coordinator passes absolute `position_ids`), so the stage must apply RoPE at the
   global position, not the local layer index. (Layer index ≠ position — RoPE depends on position
   only, so renumbering layers is safe; confirm no layer-index-dependent term exists in Qwen2.)

## Stage load path (next increment, after the slicer validates)
`stage_worker` gets a mode: given a sub-model dir + the global `(start,end)` it covers, load via
`load_model` (device_map=cuda → AWQ/Marlin), set `embed_tokens=Identity`, and run its `n` local
layers on the incoming activation, mapping the global slot to local `0..n-1`. Gated by env
(e.g. `CIRCUIT_STAGE_SUBMODEL=<dir>`); the bnb shard path stays the default until AWQ validates.

## Fallback if AWQ-per-node doesn't pan out
A fat-node **bnb** mesh (fewest-fattest already done) stays decentralized and correct, just
~2.1× slower per node. Acceptable interim; AWQ-per-node is the upgrade, not a prerequisite for a
working mesh.

## Productization — the contributor on-ramp (BUILT, GPU/HF validation pending)
Two pieces turn the validated static path into something a contributor can actually run in the
**dynamic** mesh:

**#1 Per-stage artifact distribution** (`scripts/publish-awq-shards.py`): a one-time job slices the
full AWQ into the canonical pipeline layout — coordinator keep-head slice + per-stage slices — and
publishes them to ONE private Circuit-LLM HF repo (one subfolder per slot) plus a `manifest.json`.
So a node pulls only its **~16GB slice**, not the full 40GB. Get the bandwidth-proportional layout
for a target fleet from `python3 -m engine.topology layout <N> <gpu0> <gpu1> …`, pass it as
`--layout 0:59,59:80`. Pure `parse_layout`/`build_manifest` unit-tested (`test_shard_fetch`).

**#2 Dynamic-mesh wiring** (`engine/shard_fetch.py` + `run_control_client`): a node joins via
`--control-url`, the coordinator assigns it `[start,end)`, and `resolve_submodel(model, start, end,
repo=…)` obtains the AWQ sub-model for *exactly that range* — **local cache → download the published
artifact (#1) → slice locally from a staged full checkpoint** — then serves it with Marlin. Enabled
by `CIRCUIT_AWQ_SHARDS=<repo>` (or `=local` to always slice). Unset → the original bnb/shard path.
This is what makes a contributor's node run AWQ (2.1× bnb) instead of bnb in the live mesh.

**Validation pending (next GPU/HF session):** publish a real shard repo + a dynamic node that pulls
its slice and serves; confirm the assigned range aligns with a published slot (catalog-aligned slot
boundaries) or falls back to local slicing cleanly. **Scaling = REPLICATION** (multiple pipelines)
— each L40+L4-class pipeline caps ~16 tok/s aggregate, so capacity grows by adding pipelines, not
by tuning one (measured 2026-06-23).
