# Floating Coordinator — removing the single-coordinator funnel

**Status:** SPEC (not built). The design we build against; do it right, not fast.
**Companion to:** `docs/SCALING.md` / `docs/SPEED_ROADMAP.md` (replication is the proven throughput
lever), `docs/TOPOLOGY_AWARE_ROUTING.md` (routing foundation), `docs/VERIFICATION.md` (trust/ban),
`docs/AWQ_PER_NODE.md` (per-node slices), `docs/BATCHING.md` (overlap/intra-step).
**Constraint (non-negotiable):** every part of this must hold on a *randomly scattered, high-RTT
(~80–150 ms/hop) mesh*. No co-location, no assuming nodes are nearby, no topology we hand-pick.
Validate on scattered nodes only; co-located benchmarks are diagnostic, never the target.

---

## 1. Problem

The coordinator is a single point of *throughput* (and of failure). It holds the only **head**
(embedding + final norm + lm_head + the 1.5B draft) and **orchestrates every session**: it builds
the draft tree, dials the pipeline, verifies, samples, streams — for all users.

Replication (`Registry.acquire_route`, `docs/SCALING.md`) lets contributor GPUs add *parallel
pipelines* — R replicas of a slot → ~R lanes → aggregate throughput scales with the network. That
is the decentralized net's real metric (measured ~3–4× at concurrency 4). **But every one of those
lanes funnels through the one coordinator.** So adding replicas eventually stops helping: you're
capped by one node's head-compute + draft-compute + orchestration fan-out. Our own notes flagged
this as the next bottleneck after replication: *"multi-coordinator later."*

This doc is "later." It is **the unlock for replication to actually scale**, and it removes the
coordinator SPOF as a consequence.

## 2. Measured grounding (read before designing)

- **Single-stream is physics-bound on a scattered net.** 4× L4 prod: ~7.9 tok/s single, accept
  0.597 / **3.39 tokens per round-trip**. The 1.5B draft already sits near EAGLE-3's published
  accept band — predictive drafting is what makes a high-RTT pipeline tolerable, and it is close
  to maxed. **Do not expect single-stream gains here. Keep the 1.5B draft; do not chase EAGLE.**
- **Throughput is the metric, replication is the lever.** Proven ~3–4× via pipeline overlap; the
  notes are explicit that *replication (more lanes), not kernel/scheduler tuning, is the scaling
  lever.* The coordinator funnel is what stops that lever from scaling indefinitely.
- **Chain relay is NOT a dependency.** Its win needs network-clustered nodes we don't control
  (chain ≈ star on scattered placement, measured). Floating coordinator must work over the
  existing star. Chain stays gated, orthogonal.

**Therefore:** the floating coordinator is a *throughput-scaling + fault-tolerance* change, not a
single-stream change. Selling it as single-stream speed would be stacking the deck.

## 3. Goals / Non-goals

**Goals**
- Any node can orchestrate any session → no single head/orchestrator funnel or SPOF.
- Aggregate throughput scales ~linearly with (orchestrators × replicas) as scattered GPUs join.
- An orchestrator death does not lose in-flight conversations (cheap re-home, no full re-prefill).
- Preserve the predictive-drafting (tree) path byte-for-byte; preserve verification/ban/stake.

**Non-goals**
- Single-stream latency improvement (it's physics-bound; the draft already amortizes the hop).
- Changing the model/quantization/slice scheme (`docs/AWQ_PER_NODE.md` stands).
- Geo-clustering or co-location of any kind (forbidden).
- Multi-coordinator *consensus on the hot path* — the control plane stays off the per-token path.

## 4. The split: control plane vs data plane

The coordinator does two unrelated jobs. Separate them.

### 4a. Control plane (the brain) — singular, lightweight, off-GPU-capable, HA-able
The registry/router: slot→holders, replicas, RTT matrix, per-holder load, stake/trust, bans. Pure
logic, ~no compute, **already persisted** (`CIRCUIT_REGISTRY_STATE`). It answers two questions:
- `acquire_entry()` — which **orchestrator** should a new session use (least-loaded, optionally
  RTT-near the client).
- `acquire_route(session)` — a route through the **slice replicas** (existing; least-loaded
  holder per slot, RTT tie-break, KV-affinity pin).

It is **not on the per-token path** — an orchestrator calls it once at session start (and on
failover). A single instance is fine to launch; HA (standby/Raft) is Phase 4, since downtime only
pauses *new* sessions. It can run on a stable host (the VPS) or a small pod — no GPU needed.

### 4b. Orchestrator role (the head) — replicated onto every GPU node
Each node also loads the **head bundle**: embedding + final norm + lm_head + **the 1.5B draft +
tree-draft loop, unchanged**. ~4 GB VRAM. Any node can be a session's entry point and run the full
predictive-drafting loop against a route it gets from the control plane.

**Best-practice decision — head-only orchestrator (no co-located layer slice).** Today the
coordinator co-locates layers `[0,20)` to save one hop. Giving that up costs +1 hop/round but buys
what matters at scale:
- Orchestrators are **light (~4 GB) and interchangeable** — any serves any session.
- Head can run on a **small/cheap card** (or shared), so orchestration capacity is cheap to add.
- **All 80 layers' KV lives on slice-holders** → orchestrator is ~stateless → its death doesn't
  lose the conversation (§6).

The +1 hop is acceptable *because* single-stream is already physics-bound and the draft amortizes
the round-trip — we trade a hair of latency for linear throughput scaling and clean failover. A
latency tier *may* still co-locate a slice on its orchestrator (per-node config, not a fork).

### 4c. Slice holders — unchanged
Layer-range workers (`docs/AWQ_PER_NODE.md`), replicated, serving any orchestrator's verification
passes keyed by **session-id** in their shared KV cache. No change beyond §6's attach/free hygiene.

## 5. Request flow (predictive drafting preserved)

```
client → gateway → control plane.acquire_entry() → orchestrator O
O: embed(prompt)                         # prefix cache covers the system-prompt KV
O: control plane.acquire_route(session)  # [replica of slot0..slotN], RTT/load-balanced, pinned
loop until EOS/limit:
  O: 1.5B draft builds a TREE of K candidates           # local, cheap, distributed across O's
  O: send tree (tokens+positions+tree-mask) THROUGH the route — verify all candidates in ONE pass
  slices: run owned layers on the tree batch, keyed by session-id (warm KV)
  O: norm + lm_head + verify tree → accept longest agreed path, sample next, stream
  # ~3.3 accepted tokens per network round-trip (measured); draft compute now spread over all O's
O: control plane.release_route(session)  # on end/reset/failover
```

Because each orchestrator runs **its own** draft, draft compute — today all on the one coordinator —
**distributes across the network**. That's real throughput headroom on top of removing the funnel.

## 6. KV, session affinity, and failover (the elegant part)

The orchestrator holds **no layer KV** (it's all on slice-holders, keyed by session-id). So:

- **Orchestrator dies mid-session:** control plane re-homes the session to another orchestrator,
  which re-`acquire_route`s the *same* route and **re-attaches to the existing per-session KV on
  the slice-holders — no re-prefill**, the conversation continues. (Head-side state — the running
  token sequence — is small; carry it in the session record or have the client resend on re-home.)
- **Slice replica dies:** `acquire_route` picks another replica for that slot (needs replication
  ≥ 2); only that slice re-prefills (bounded; prefix cache covers the system prompt).
- **Control plane dies:** in-flight sessions keep streaming (route already pinned client/orchestrator
  side); only *new* sessions pause until it's back (persisted; HA in Phase 4).

No single death is a global outage. Full fault tolerance falls out of the throughput design rather
than being bolted on. **Required hygiene (already flagged):** session-keyed KV free on
end/reset/failover (`CHAIN_KV_CTRL`/free-on-end) so re-homing and lane churn don't leak KV.

## 7. Mapping to existing code (these seams already exist)

This is mostly a refactor — the engine was built with the joints:
1. **Extract the head as a standalone `Orchestrator`.** `coordinator.py` already isolates
   embed/lm_head/draft + `request_gate`/`_Conn`/`_ConnPool` and the spec loop
   (`generate_speculative_stream`). Wrap head+draft+loop to take a *route* from the control plane
   instead of owning the topology. Keep `CIRCUIT_DRAFT` (1.5B) as-is.
2. **Promote the registry to a standalone control-plane service.** `registry.py` + `control_server.py`
   are already a clean, persisted object. Add `acquire_entry()` beside `acquire_route()`; run as its
   own process; orchestrators + gateway call it over the control channel.
3. **Expose `acquire_route`/`release_route` on the control channel** so remote orchestrators get
   routes; session-load accounting (`_load`/`_session_load`) moves server-side (already there).
4. **Gateway picks an orchestrator per request.** `circuit-data-api/inference-gateway.js` currently
   always hits `:19200` (the one coordinator); change it to `acquire_entry()` then proxy there.
   Stateless, trivial.
5. **Session-id attach path on `stage_worker`.** It already keys KV by session-id in a shared cache;
   add an explicit "attach to existing session" entry so a re-homed orchestrator resumes without
   re-prefill, plus the free-on-end hygiene from §6.
6. **Star relay stays; chain stays gated** (§2). No dependency on clustering.

## 8. What to expect (honest)

| Dimension | Effect | Why |
|---|---|---|
| Single-stream latency | ≈ unchanged (≈ −1 hop if head-only) | physics-bound; draft already at ~3.3 tok/round |
| Aggregate throughput | **scales ~linearly with (orchestrators × replicas)** | no coordinator funnel; draft compute distributed |
| Resilience | **no SPOF** (head or slice can die) | head-only orchestrators + KV-on-slices re-home |
| VRAM cost | +~4 GB per node (head bundle) | embed+norm+lm_head+1.5B draft |

## 9. Build order (do it right; byte-identical gates like every prior engine change)

1. **Control/data-plane split, behavior-preserving.** Registry as a standalone service; one
   orchestrator still in the loop. DoD: output **byte-identical** to today (the standing engine
   gate), `coverage_ok`, no perf regression.
2. **Head-only orchestrator + remote `acquire_route` + gateway `acquire_entry`.** Still one
   orchestrator. DoD: parity/byte-identical; failover re-home re-attaches KV (no re-prefill) in a
   unit test over a synthetic latency matrix (`topology.py` is pure logic).
3. **Second orchestrator + session re-home on the shared replica pool.** DoD: measure aggregate
   throughput scaling on a **scattered** mesh (the only valid test); confirm draft compute
   distributes; orchestrator-kill mid-session continues the conversation.
4. **Control-plane HA** (standby/Raft) — last, off the hot path.

## 10. Risks / open questions

- **Routing consistency under churn.** Two orchestrators must not double-book a replica's capacity
  → load accounting must be server-side + atomic in the control plane (it already centralizes
  `_load`); revisit lock granularity.
- **Head-side session state on re-home.** Decide where the running token sequence lives (session
  record vs client resend). Smallest-surface option: client resends the tail; prefix cache + slice
  KV make it cheap.
- **Verification/trust across orchestrators.** Bans/eviction counts are control-plane state
  (persisted) → already shared correctly; confirm the auditor still pairs probation↔trusted holders
  independent of which orchestrator drove the work.
- **Entry selection policy.** Least-loaded vs RTT-near-client vs stake-weighted — start least-loaded,
  make it pluggable.
- **Backpressure.** A flooded orchestrator must shed to the control plane's next pick, not queue
  unboundedly.

## 11. One-line summary

Split the coordinator into a **singular off-GPU control plane** (routing/trust, persisted, HA-able)
and a **head bundle replicated onto every GPU** (embed/lm_head/**1.5B tree draft**). Any node
orchestrates any session against a control-plane route; layer KV lives on the slice-holders so an
orchestrator death re-homes without re-prefill. Result: replication finally scales (no funnel),
the SPOF is gone, the tree draft and star relay are untouched — and none of it assumes co-location.
