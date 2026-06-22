# Circuit DLLM — 2× L4 WAN Pipeline (decentralized split)

> Concrete design: a single model split across **two separate RunPod L4
> instances over the open internet**, served fast via async-pipelined
> speculative decoding. This is the leyten/shard *approach* (not its code)
> sized to our hardware. Goal: prove a real decentralized WAN split at usable
> speed, then generalize to N scattered GPUs.

---

## 0. Why this is the right test

- A **32B** model does **not fit one L4** (~19 GB weights + KV + overhead > 23 GB
  usable) but **fits across two** → a *genuine* split, not an artificial one.
- Two RunPods in **different locations** = a real WAN topology, not co-located
  cheating. If it's fast here, the architecture generalizes to scattered GPUs.
- Only **2 stages / 1 WAN hop** → the simplest possible pipeline to get async
  speculation right before scaling to 6+ stages.

**Target model:** Qwen2.5-32B-Instruct, 4-bit (AWQ/GPTQ), 64 layers split 32/32.
**Draft model:** Qwen2.5-3B-Instruct (CUDA-graphed) on the coordinator.

---

## 1. Topology

```
        ┌─────────────── RunPod A (L4 #1) ───────────────┐
 user → │  COORDINATOR (co-located w/ stage0)            │
        │   • token embedding + lm_head + sampling       │
        │   • 3B draft model (CUDA graph)                │
        │   • async pipeline scheduler + spec accept     │
        │   • API endpoint (OpenAI-compatible)           │
        │  STAGE 0:  layers  0–31   (~9.5 GB)            │
        └───────────────────┬────────────────────────────┘
                            │  ONE WAN hop (encrypted activations)
                            ▼
        ┌─────────────── RunPod B (L4 #2) ───────────────┐
        │  STAGE 1:  layers 32–63   (~9.5 GB)            │
        │   • per-stage KV cache w/ speculative rollback │
        └───────────────────┬────────────────────────────┘
                            │  direct return to coordinator (1 hop)
                            ▼  (argmaxes/logits, not relayed back through stage0)
                       COORDINATOR
```

- **Coordinator holds no transformer layers** of the split model — just
  embedding/head + the draft. It is **co-located with stage 0** so the only WAN
  crossing is stage0 ↔ stage1 (out) and stage1 → coordinator (return).
- **Ring/direct-return:** stage 1 returns straight to the coordinator instead of
  relaying activations back through stage 0. For 2 stages that's **1 hop out +
  1 hop back = one WAN round trip per pipeline traversal** — which async
  pipelining then hides.

---

## 2. The decode loop — async-pipelined speculative decoding

This is the whole game. It converts a **latency-bound** loop into a
**throughput-bound** one.

```
repeat:
  1. DRAFT  — 3B proposes K candidate tokens (CUDA graph, ~13 ms)
  2. EMBED  — coordinator embeds [positions + K proposals]
  3. STAGE0 — forward layers 0–31  → activations
  4.   ⇒ WAN ⇒  send activations to stage1
  5. STAGE1 — forward layers 32–63 → argmaxes
  6.   ⇒ WAN ⇒  direct-return to coordinator
  7. ACCEPT — lm_head + greedy acceptance → commit matching prefix (A tokens)
  8. ROLLBACK — advance each stage's KV by A; discard rejected speculative KV

KEY: do NOT wait for step 7 before starting step 1 of the next chunk.
     Launch chunk N+1 speculatively (off the predicted continuation) while
     chunk N is still in the pipeline. Keep ~2–3 chunks in flight so the
     WAN hop is overlapped with useful work. "The loop runs at the pipeline's
     throughput, not its latency."
```

- **Speculation provides the independent work** that fills the pipeline — that's
  what makes async pipelining possible despite autoregressive dependency.
- **Misprediction penalty** = discard the in-flight chunks past the rejection
  point and refill. Cheap relative to the WAN we're hiding.
- With good acceptance + 2–3 chunks in flight, the WAN drops to a small fraction
  of the loop and throughput ≈ the **slowest stage's rate**.

---

## 3. Best-practices checklist

**Compute**
- [ ] **CUDA graphs** on the draft (shard saw 3.8×: ~50 → ~13 ms/token) and on
      each stage's forward — kills per-step kernel-launch overhead.
- [ ] **Static-address position/KV tensors** so spec rollback stays byte-identical
      to the eager path under graph capture.
- [ ] **4-bit weights** (AWQ/GPTQ) for the 32B; keep activations fp16 on the wire.
- [ ] Per-stage **KV cache keyed by session** with explicit "accept M of K /
      rollback K−M" support.

**Network (over the open internet)**
- [ ] **Authenticated encryption** on every frame (ChaCha20-Poly1305), length-
      prefixed binary framing, **no pickle/JSON in the hot path**.
- [ ] **Persistent connections + `TCP_NODELAY`** (disable Nagle — our small frames
      are exactly what it stalls). Consider QUIC for head-of-line avoidance.
- [ ] **Activation quantization** (fp8/int8 on the wire) — optional; we're
      latency-bound not bandwidth-bound, so do this only if it's free.
- [ ] **Edge RTT health monitoring** — measure stage↔stage RTT continuously;
      surface it; use it to tune in-flight chunk count.

**Decentralization / trust**
- [ ] **Verifiable run receipts** — GPU UUIDs, public IPs, regions, measured RTTs,
      output token hash, sync-vs-pipelined match — for payment + anti-cheat.
- [ ] **Graceful stage failure** — a dropped stage fails the in-flight request and
      reroutes; reconnect without crash-looping (bounded backoff).

**Placement**
- [ ] **Start same-region** (RTT ~10–20 ms) to validate correctness + the async
      machinery, *then* move to cross-region to measure WAN-hiding honestly.

---

## 4. Stack note (honest)

Our current coordinator is Node + a custom C runner. The best-practice
implementation of this — CUDA graphs, AWQ/GPTQ kernels, graph-captured spec
rollback — strongly favors **Python + PyTorch** for the stage workers and draft.
Recommendation: **build the stage workers + draft in Python/PyTorch**; the
API/routing/payment layer can stay in our existing stack and talk to them over
the encrypted wire. This is a real rebuild of the compute path, not an extension
of the C runner — budget for that.

---

## 5. Estimated speed (derivation)

Throughput ceiling (async, WAN hidden) ≈ **1 / (slowest stage time)**.

- Each L4 stage holds 32 layers of the 32B ≈ **9.5 GB**.
- L4 memory bandwidth ≈ **300 GB/s** → weight read ≈ 9.5 / 300 ≈ **32 ms**.
- \+ attention/sampling overhead (~1.2×) ≈ **~38 ms/stage**.
- → stage rate ≈ **~26 passes/sec**.

| Implementation level | What's on | Est. tok/s (32B) |
|---|---|---|
| Naïve synchronous | one token per WAN round trip | **~3–8** (worse the farther apart) |
| \+ speculative decoding only | amortize hop over A tokens | ~10–20 |
| **\+ async pipelining (full design)** | WAN hidden, throughput-bound | **~20–30** |

**Headline: ~20–30 tok/s for a 32B across 2 WAN L4s** with the full async design;
conservatively plan **~20–25**. TTFT ~0.5–1.5 s (one prefill traversal).

**Sanity check vs shard:** their L4-equivalent stage time is ~34 ms (62 GB at
1.8 TB/s on a PRO 6000) and they hit ~30 tok/s — i.e. tok/s ≈ stage rate. Our
stage time (~32–38 ms) is *comparable*, and our 2-stage pipeline is *shorter*
(easier to keep full, fewer hops to hide) than their 6-stage — so ~20–30 tok/s
is a defensible target, not optimism.

**What moves the number**
- **RTT/placement** — same-region (~10–20 ms) needs fewer in-flight chunks and
  wastes less speculation than cross-country (~70 ms) or intercontinental
  (~140 ms). Async hides the hop, but lower RTT still helps.
- **Acceptance** — code/repetitive content → top of range; hard/creative → bottom.
- **Implementation quality** — CUDA graphs and a tight async scheduler are the
  difference between the ~10 and the ~30 row. This is where the eng effort lands.

---

## 6. Build phases

1. **Correctness, co-located:** 2 L4 stages same-region, *synchronous* spec
   decode, encrypted wire, per-stage KV + rollback. Verify output is byte-exact
   vs a single-GPU reference. (Expect ~10–15 tok/s here — that's fine.)
2. **Async pipeline:** add multi-chunk-in-flight scheduling + CUDA graphs.
   Measure the jump toward ~25–30 tok/s same-region.
3. **Go WAN:** move stage 1 to a different region. Measure how much speed holds
   (the real test of latency-hiding). Tune in-flight depth to the RTT.
4. **Decentralize:** receipts, payment split, dynamic join/leave, then generalize
   the 2-stage pipeline to N scattered stages.

---

## 7. Bottom line

Two scattered L4s can serve a **32B at ~20–30 tok/s over WAN** — *if* we build
async-pipelined speculative decoding properly (CUDA graphs, ring return,
encrypted wire, tight in-flight scheduling). That's a genuine decentralized split
of a model no single L4 could hold, at a usable speed. The speed ceiling is the
L4 stage rate (~26/s); the engineering is what closes the gap from the ~5 tok/s
naïve floor up to that ceiling.
