# Bigger Model — Holistic Review & Plan

> **End goal:** run a model too big for one L4 (a 70–72B, 4-bit ≈ 40 GB) across 3–4 L4s,
> each GPU loading **only its own layers** — which is the point at which the mesh stops
> being a demo (32B fits one card) and becomes genuinely necessary.

---

## 1. What already works (the ~90%)

Everything except one thing:
- **Distributed pipeline inference** (coordinator → stages, encrypted wire) — live.
- **Mesh**: dynamic join, slot assignment by capacity, ed25519-signed registration + enforcement, self-discovery, dedupe — all proven on real L4s.
- **Unified node-client install** (image + 2 env vars, no SSH) — proven.
- **Predictive drafting** with the 0.5B — works for any Qwen2.5 target, *including a 72B* (same family/tokenizer), so no new draft needed.
- **Topology** already supports the coordinator co-locating a slice + N nodes covering the rest.

So a 72B needs **no new distributed/mesh/auth work**. The model just has to *load*.

## 2. The one blocker: selective shard loading for a QUANTIZED model

A stage must load **only its layers** into VRAM (never the whole model). For an
**un-quantized** model this is done and bitwise-proven (`load_model_shard`). For a
**4-bit (AWQ/GPTQ)** model — which is the only way a 72B fits L4s — both attempts fail:

| Approach | In code | Why it fails |
|---|---|---|
| Dense skeleton (`load_model_shard`) | wired into `--shard` | `from_config` builds plain `Linear`; AWQ `qweight`/`scales`/`qzeros` don't bind → weights silently "not used" (verified today: 3.12 GB VRAM where ~12 GB expected, garbage output) |
| `device_map` w/ un-owned on `meta` (`_shard_device_map`) | **defined, never called** | gptqmodel's Marlin `post_init` calls `get_device_properties` on **every** layer → `ValueError: Expected a cuda device, but got: meta` |
| Current prod: load-then-prune (`prune_to_layers`) | live | peak load = the **whole** 4-bit model on one card → caps at ~32B; AWQ prune doesn't even free VRAM |

**Root cause:** the quant loaders (gptqmodel/Marlin) assume the entire quantized model
is constructed and resident on one GPU. Sharding requires constructing/initializing
**only the owned layers** without the loader choking on the absent ones.

### ⚠️ Process issue found
The shard test (`tests/test_shard_load.py`) asserts only **shape + VRAM**, not
**correctness** — that's how it reported "SHARD LOAD OK" on a model whose weights never
loaded. **Any fix must be validated by output correctness vs a reference forward**, and
that test must be upgraded first so it can't lie to us again.

## 3. Options (ranked by likely effort-to-working)

### A. bitsandbytes NF4 + accelerate `device_map`  *(most likely to just work)*
bnb quantizes from the fp16 checkpoint **at load**, and accelerate's `device_map` shards
that load across devices — no Marlin `post_init`, so `meta`/per-GPU placement is
supported (well-trodden path). Our own notes: bnb prune **does** free VRAM (AWQ doesn't).
- **Pros:** likely small integration; standard accelerate; sidesteps the Marlin wall.
- **Cons:** needs the **fp16** checkpoint (~140 GB for 72B) downloaded per node (~30 min, big disk); quality/speed a touch below AWQ-Marlin.

### B. Quant-aware AWQ skeleton  *(keeps AWQ; real engine work)*
Build the skeleton with gptqmodel **QuantLinear** for owned layers (so the quantized
tensors bind), `Identity` the rest, load only owned tensors, run Marlin `post_init`
**only** on owned layers.
- **Pros:** stays on AWQ (40 GB download, best quality/speed); the "right" answer.
- **Cons:** threads through gptqmodel internals (construction + packing + post_init); highest effort + fragility.

### C. Patch/monkeypatch gptqmodel `post_init` to skip non-owned layers  *(targeted hack)*
Wrap gptqmodel so `post_init`/Marlin only iterates owned (cuda) layers.
- **Pros:** could unlock the `device_map`+`meta` path cheaply.
- **Cons:** depends on gptqmodel version internals; brittle across upgrades.

## 3b. The un-quantized shortcut (prove it NOW with done code)

Key correction worth stating: **un-quantized is BIGGER, not smaller.** fp16 = ~2 bytes/param,
4-bit ≈ 0.55. So fp16 32B ≈ **65 GB** vs AWQ 32B ≈ 19 GB. An un-quantized 32B therefore
needs **3–4 L4s** (more than today's 2), not fewer.

But the angle is still excellent, because **the un-quantized selective loader is already
built and proven** (`load_model_shard`, bitwise on CPU). So we can:
- Run **fp16 Qwen2.5-32B (~65 GB) sharded across 3–4 L4s right now** — a model that genuinely
  *doesn't fit one card* — with **no quant fix**, and at **higher quality** than the AWQ-32B
  the live site runs today.
- This **exercises the entire too-big-sharded pipeline** (shard-load → mesh → correctness)
  end-to-end on real GPUs, **de-risking everything before** the hard quant work.

**Does it generalize?** Yes — the dense loader is architecture-general, so it shards *any*
un-quantized model (fp16 70B would shard the same way, just across ~7 L4s). And the quant
loader, once fixed, shards *any* quantized model. Neither is 32B-specific. **The catch:** the
un-quantized loader does **not** unlock the *quantized* path — "big model on FEW GPUs" still
needs the §3 fix, because an fp16 70B (~140 GB) needs ~7 L4s, defeating the point.

**So:** un-quantized 32B = a real, near-term win on done code (proves the mesh's value +
a quality bump), and a clean stepping-stone; quantized 72B = the efficiency end-goal that
still needs the §3 work.

## 4. Recommended plan (phased, spike-first)

**Phase 0 — Prove the too-big-sharded pipeline with the un-quantized 32B (done loader).**
Validate `load_model_shard` on a GPU (it's only CPU-proven), upgrade the test to check
**correctness**, then run fp16 32B across 3–4 L4s via the mesh and confirm coherent
inference. Low risk, uses existing code, immediately demonstrates "a model too big for one
card, served across the mesh" — and surfaces any GPU-side shard-load issues before §3.


**Phase A — Spike to find the working mechanism (cheap, 1 GPU, small model first).**
1. Upgrade `test_shard_load.py` to assert **correctness** (sharded stage forward == full-model forward for those layers, within fp tolerance), not just shape/VRAM.
2. Spike **Option A (bnb)** on a small model (e.g. a 7B) on one L4: `device_map` with un-owned on `meta`, confirm only owned layers materialize **and output is correct**. If good → bnb is the path.
3. If bnb is unsuitable (download size / quality), spike **Option B** (QuantLinear owned-only) on the same small model.
   → **Exit:** one approach that loads only owned layers AND is provably correct.

**Phase B — Implement the chosen loader.**
- Wire it behind the existing `--shard` flag (replacing/ුamending `load_model_shard` for quant), keep dense path for non-quant. Coordinator uses it with `keep_head=True` for its co-located slice.

**Phase C — Validate a 72B across L4s (off to the side, not prod).**
- `Qwen/Qwen2.5-72B-Instruct-AWQ` (or the bnb equivalent), 80 layers. Coordinator (ends + 0:20) + 3 nodes (20:40/40:60/60:80). Confirm each L4 holds only its slice (~10–14 GB), and **inference is coherent** end-to-end (+ predictive drafting via the existing 0.5B).

**Phase D — Make it the live mesh (the migration).**
- Persistent coordinator + stable control address; nodes join; repoint the website; retire the 32B static pair. (This is the "go live" plan from the unified-node work, now serving the 72B.)

## 5. Honest sizing & risks
- **This is the real engineering of the project.** Phase A is a few focused sessions of spikes; B+C depend on which option wins (bnb likely days; AWQ-skeleton likely longer). Not a config flip.
- **Risk:** even the "easy" bnb path has the 140 GB fp16 per-node download; the AWQ path has gptqmodel-internals fragility. The spike de-risks the choice before we commit.
- **Hard dependency on correctness validation** — quantized + sharded is exactly where silent wrongness hides (today's false positive proves it).
- **Capacity:** needs 3–4 L4s online to cover a 72B; EUR-IS-1 has been L4-constrained — provision where capacity exists.

## 6. Target
**Qwen2.5-72B-Instruct** (4-bit, ~40 GB) across **4 L4s** — same family as today's 32B,
so the engine, tokenizer, and 0.5B draft all carry over. That's the concrete model to
aim the whole plan at.
