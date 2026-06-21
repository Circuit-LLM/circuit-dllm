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

## 3. Win B — intra-step batching (Phase 2 sketch)

The real rewrite, recorded so Win A is built pointing the right way:

1. **Batched wire.** ACTIVATION carries `[B, T, D]` for B sequences; RESULT likewise.
2. **Ragged KV per stage.** Sequences sit at different positions/lengths. Either pad +
   attention-mask, or **paged attention** (block KV) for memory efficiency — paged is
   the right long-term answer (no fragmentation, supports large B).
3. **Continuous-batching scheduler** (coordinator): one loop that each step admits new
   requests (prefill), runs a batched decode for all active sequences, and evicts
   finished ones. Prefill vs decode shapes differ — separate phases or chunked-prefill.
4. **Pipeline × batch (the hard part).** Micro-batch / 1F1B scheduling so both pipeline
   stages stay busy with the batch instead of bubbling. This is where pipeline-parallel
   batching is genuinely harder than single-GPU batching.

High value, high risk, multi-week, from-scratch (no vLLM to lean on). Gated behind the
same `max_concurrency` / a batch-size knob; same token-identical correctness contract.

---

## 4. What needs the live hardware (can't be settled on CPU)

- **Bottleneck measurement (do first).** Is 11 tok/s pipeline-bubble-bound (→ overlap
  buys a lot) or genuinely compute-bound (→ overlap buys little, go straight to B)?
  A short profiling pass on the 32B: per-stage time, the idle gap per token, prefill vs
  decode split. This number decides how much to invest in A vs B.
- **Throughput + memory validation.** Real tok/s under concurrent load and the per-session
  KV footprint that sets `max_concurrency` — only measurable on the L4s.

---

## 5. Rollout (and why this order)

0. **Design (this doc) + a profiling deploy** to measure the real bottleneck.
1. **Win A — overlap.** Connection pool + concurrency semaphore + thread-safe state,
   gated `CIRCUIT_MAX_CONCURRENCY` (default 1 = byte-identical), CPU correctness test.
   Deploy, raise concurrency cautiously, measure.
2. **Win B — intra-step batching.** Batched wire + ragged/paged KV + continuous
   scheduler. Deploy, measure, tune batch size against memory.

**Invariant throughout:** default config = today's behaviour, byte-identical; output is
always token-identical to sequential greedy; the live engine changes only on a
deliberate, measured deploy. Same discipline as the mesh work — gated, regression-tested,
never a silent change to the running system.
