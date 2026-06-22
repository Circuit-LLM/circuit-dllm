# Circuit DLLM — Concurrency & Batching

How the single-stream engine becomes a multi-request server — so it can serve a
swarm and amortise GPU cost (the lever that makes a per-call price real). Written
before touching the hot path, holistically, so the increments are deliberate.

---

## 0. Where we are

The engine serves **one request at a time**. The HTTP server is already threaded
(`ThreadingHTTPServer`), but both generation paths wrap the whole decode in a single
global `with _lock:` (`engine/api.py`), so requests serialise. Measured ~11 tok/s on
the 32B split across 2 L4s, with **each GPU ~40–50% idle** in the sequential pipeline
(pod1 = embed + co-located stage + head; pod2 = the tail stage).

That idle is the opportunity, and it tells us the bottleneck is **scheduling, not
flops**: the GPUs spend half their time waiting for the other half of the pipeline.

### What already supports concurrency (and what doesn't)

| Component | Concurrency-ready? |
|---|---|
| Stage worker (pod2) | **Yes** — thread-per-peer accept loop, a per-connection `sessions` dict (isolated KV), and a `compute_lock` that serialises the actual forward. Multiple coordinator connections already work. |
| Coordinator KV (`_local_caches`, `_session_routes`) | **Yes** — keyed by session id, isolated. |
| Coordinator **sockets** (`self.socks` static / `self._conns` dynamic) | **NO** — shared. Two requests writing framed messages to one socket interleave bytes and corrupt the wire. **This is the core blocker.** |
| `self._session` counter, `self._spec` metrics | **No** — racy increments (cheap to fix). |
| api.py global `_lock` | The serialiser itself. |

So the stages are ready; the **coordinator** is the single-stream chokepoint, and the
specific thing in the way is shared sockets.

---

## 1. Two distinct wins (don't conflate them)

**"Batching" hides two independent multipliers:**

### Win A — Pipeline overlap (cross-request)
Run N requests concurrently. While request X blocks on pod2, request Y uses pod1.
Both GPUs stay busy. Ceiling ≈ recovering the 40–50% idle → **up to ~2×**. Crucially,
**no stage-worker or KV change** — the stages already isolate per-connection sessions.
The work is entirely coordinator-side: give each in-flight request its own sockets,
make the shared counters thread-safe, and bound concurrency. **Smaller, lower-risk,
correctness-testable on CPU. This is the first increment.**

### Win B — Intra-step batching (multi-sequence per forward)
Pack N sequences' tokens into ONE forward per stage. GPU matmuls are batch-efficient,
so throughput scales near-linearly until memory/compute saturate — the vLLM-class win,
**the big multiplier**. But it's a real rewrite: batched activations on the wire,
ragged/padded (or paged) KV per stage, and a continuous-batching scheduler. **Phase 2.**

Win A de-risks the path to Win B (the connection/concurrency plumbing is shared) and
buys time while B is built.

---

## 2. Win A — pipeline overlap, in detail

### Design
1. **Connection pool.** A request "checks out" a *connection set* (one socket to each
   stage, dynamic: one per routed holder) for its lifetime and returns it on finish.
   Pool size = `max_concurrency`. Concurrent requests never share a socket, so the
   framed protocol is safe. The stages already give each connection its own KV space,
   so per-request sockets ⇒ per-request KV isolation on the stage for free.
2. **Bounded concurrency.** Replace api.py's global `_lock` with a
   `Semaphore(max_concurrency)`. **`CIRCUIT_MAX_CONCURRENCY` defaults to 1 → a
   semaphore of 1 → byte-identical to today's single-stream.** Overlap is opt-in; the
   running engine is untouched until the operator raises it.
3. **Thread-safe shared state.** Atomic session ids (`itertools.count`); guard `_spec`
   updates with a tiny lock (observability only — never on the compute path's result).
4. **pod1 self-serialises correctly.** Concurrent threads calling embed/local/head run
   on one GPU; the GIL + CUDA serialise them. That's the *right* behaviour (one GPU)
   — the win is overlapping pod1 with pod2, not parallelising pod1 with itself.

### Correctness contract
Overlap changes *scheduling, never results*. N concurrent `generate()` calls must each
be **token-identical** to their own sequential greedy reference. Tested on CPU with 2
stage workers and concurrent threads; the throughput win itself is measured on the L4s.

### Bounds & caveats
- **Memory.** N concurrent requests ≈ N× KV on the stages + N× local KV. For the 32B
  this is real VRAM; `max_concurrency` is the safety bound. Size it from measured
  per-session KV footprint.
- **Speculative path.** `SocketTarget` relays through the same pool, so overlap applies
  to the live speculative path too.
- **Prefill heads-of-line.** A long prompt's prefill holds pod1 for a while; overlap
  softens but doesn't eliminate it — Win B's chunked prefill does.
- **Failover interaction.** The pool must integrate with the dynamic failover/re-prefill
  (per-request route pin + per-request sockets); the recently-hardened fast-fail connect
  and coverage-gap paths apply per checkout.

---

## 3. Win B — intra-step batching (full design)

### 3.1 The shift
Win A runs **B independent pipelines** — B sequences, B sockets, B separate forwards;
it fills the pipeline *bubble* but every forward still processes one sequence, so the
GPU does B small matmuls. Win B runs **one pipeline over a batch** — the active
sequences are packed into a single `[B, …]` forward per stage, so the GPU does one
large, efficient matmul. That's the vLLM-class throughput multiplier; its ceiling is
GPU compute/memory, not pipeline idle. The two compose (batch the sequences, overlap a
second batch to hide the bubble) but Win B is where the throughput lives.

### 3.2 Why it's hard in THIS engine
- **Per-session KV.** `StageKV` (a HF `DynamicCache`) holds ONE sequence's K/V.
  Batching needs a cache holding B sequences whose lengths differ (one at token 12,
  another at 200) and whose membership changes every step (sequences finish, new admit).
- **Ragged decode.** Each active sequence emits one token → the batch is `[B, 1, D]`,
  clean in time but ragged in KV length; attention must let each row see only its own KV.
- **Prefill ≠ decode.** A new prompt is `[1, P, D]` (P tokens at once); a decode step is
  `[B, 1, D]`. Mixing them is the continuous-batching scheduling problem.
- **Speculative × batch.** Each sequence accepts a different number of draft tokens per
  verify → the batch desyncs and KV rolls back per-row. Batched speculative is research-grade.
- **Pipeline.** The batch must traverse the stages in a stable row order each step; the
  bubble still exists per batched step (Win A overlap optionally hides it).

### 3.3 Design decisions (best-practice scoping)
1. **Padded batching first; paged later.** Hold `[B, heads, max_len, dim]` + per-row
   lengths + a 4D mask hiding padding. Correct with the **stock HF attention — no custom
   kernel**. Wastes compute on padding (∝ length variance), so it's the right MVP, with
   **paged attention** (block KV + block tables) as the efficiency follow-up. We do NOT
   write a paged kernel up front — that's the single biggest risk multiplier.
2. **Batched greedy first; speculative deferred.** Run batched plain decode. Speculation's
   edge shrinks as batch grows (large batch is already compute-bound), and batched
   speculative is a separate hard project. Keep single-stream speculative for the
   low-concurrency latency path; batched mode trades it for throughput.
3. **Separate prefill first; chunked later.** Prefill a newcomer on the existing single-seq
   path, then admit its KV row into the decode batch. Chunked prefill is a later optimization.
4. **A scheduler owns the loop.** Replace per-request `generate*` with one coordinator
   **scheduler**: holds active sequences, each step forms the decode batch, runs one batched
   pipeline pass, samples B tokens, evicts finished, admits prefilled newcomers. Requests
   enqueue and receive tokens via per-request output queues.
5. **One connection per stage in batched mode.** The scheduler is the single producer; a
   batched ACTIVATION multiplexes B sequences over one socket per stage (Win A's per-request
   sockets are for the overlap mode). Simpler, and the right model for batching.
6. **Gated, token-identical, reversible.** A mode flag (`CIRCUIT_BATCH` / `CIRCUIT_MAX_BATCH`);
   default = today. Batched output MUST be token-identical to sequential greedy per
   sequence — a mask/index bug is a *silent* wrong-output, the central correctness risk.

### 3.4 Components (new + changed)
- **`BatchKV` (new) — the crux.** K/V for B rows at `[B, heads, max_len, dim]` per layer,
  per-row length, **slot management** (free a finished row, place a newcomer's prefilled KV),
  the 4D mask + per-row `position_ids`. Generalises `StageKV`.
- **`stage_worker` (changed).** Batched ACTIVATION → gather B rows' KV → one
  `stage.forward([B,1,D], mask, positions, BatchKV)` → scatter new K/V back → return
  `[B,1,D]`. Plus a prefill/admit op.
- **wire / tensors (changed).** Batched ACTIVATION frame: header of B×(session,pos) + the
  `[B,T,D]` tensor; RESULT mirrors. KV_CTRL gains slot free/admit.
- **`Scheduler` (new, coordinator).** The continuous-batching loop (admit/prefill/decode/
  sample/evict) + per-request output queues + max-batch/memory bounds + fairness.
- **`api.py` (changed).** Concurrent requests enqueue into the scheduler and stream from
  their queue, instead of each holding a `request_gate`.
- **Tests (new).** Token-identical-to-sequential for a mixed-length batch; dynamic
  admit/evict mid-batch; prefill-then-join; back-pressure at max batch.

### 3.5 Phased plan
- **B1 — `BatchKV` + batched stage forward (in-process).** Prove a batched forward with
  ragged padded KV is token-identical to N sequential forwards, no sockets. *The
  correctness foundation; everything rides on mask/positions being exact.*
- **B2 — batched wire + stage-worker batched path.** Two-process batched decode == reference.
- **B3 — scheduler + API.** Continuous admit/decode/evict + per-request streaming; M
  concurrent requests through one batched pipeline, each token-identical.
- **B4 — overlap a 2nd batch (compose with Win A) + chunked prefill.** Hide the bubble.
- **B5 (later) — paged attention.** Block KV for memory efficiency at large B; only if
  padded batching hits memory limits. The hardest piece, a project of its own.

### 3.6 Honest size & risk
This is **the largest single build in the project — bigger than the whole node-join
system** (topology + registry + control server + failover combined):
- **B1–B3 (the MVP that actually batches):** a new `BatchKV`, a batched wire format, a
  stage-worker batched path, a scheduler + API rework — order of **~1.5–2.5k lines of core
  code plus a heavy test suite**, built incrementally. Realistically a **focused multi-week
  effort**, with **higher correctness risk than anything so far** — a silent mask/index bug
  yields plausible-but-wrong text, caught only by exact token-equality tests on the GPU
  (CPU validates correctness, never the tok/s).
- **B4:** another meaningful chunk (overlap + chunked prefill).
- **B5 (paged):** the wildcard that can balloon; deferred behind padded batching.
- **Payoff is GPU-only.** CPU proves correctness; the throughput and the right
  `max_batch` vs VRAM are measured on the L4s.

Biggest risk = B5/paged (deferred by design). Second = silent correctness in B1's
mask/positions (front-loaded as the first fully-tested milestone). Same discipline as
everything else: gated default-off, token-identical contract, incremental + tested,
never a silent change to the running engine.

---

## 4. What needs the live hardware (can't be settled on CPU)

- **Bottleneck measurement — DONE (2026-06-21).** Win A deployed and measured: **1.40×**
  at 3 concurrent (14.2 → 19.9 tok/s aggregate), single-request latency unchanged. The
  engine **is pipeline-bubble-bound**, so batching (Win B) has real headroom toward 2×+.
- **Throughput + memory validation (Win B).** Real batched tok/s and the per-row KV
  footprint that sets `CIRCUIT_MAX_BATCH` vs VRAM — only measurable on the L4s.

---

## 5. Rollout (and why this order)

0. **Design + profiling deploy** — DONE; measured Win A = 1.40×, engine bubble-bound.
1. **Win A — overlap.** Per-request connection isolation (contextvar + `_ConnPool`),
   gated `CIRCUIT_MAX_CONCURRENCY` (default 1). **DONE — live on both pods at 4, correct,
   1.40×.**
2. **Win B — intra-step batching. B1–B3 BUILT + token-identical on CPU (2026-06-22),
   gated `CIRCUIT_BATCH` / `CIRCUIT_MAX_BATCH`, NOT yet deployed.** B1 batched forward
   (2D padding mask + per-row positions, no custom kernel), B2 batched wire
   (BATCH_ACTIVATION), B3a `generate_batch` primitive (live shape: co-located +
   remote), B3 scheduler (`engine/scheduler.py`, queue → batched decode → per-request
   stream) + API (`_serve_batched`). Static batching; B4 (dynamic admit/evict, chunked
   prefill) and B5 (paged) remain. Throughput is a GPU measurement at deploy — do it
   when concurrent load justifies it (the swarm is on OpenRouter today, so there's no
   load yet; the build is banked + ready).

**Invariant throughout:** default config = today's behaviour; output always token-identical
to sequential greedy; the live engine changes only on a deliberate, measured deploy. Same
discipline as the mesh work — gated, regression-tested, never a silent change to the
running system.
