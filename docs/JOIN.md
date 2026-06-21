# Circuit DLLM — Node Join Protocol

How an independent operator runs a GPU, joins the inference mesh, serves real
traffic reliably, and earns CIRC. Designed permissioned-first, permissionless-
capable. This is the control plane the engine doesn't have yet.

---

## 0. The data plane today (holistic recap — what we're building on)

The DLLM is **pipeline-parallel**: the model is split by transformer layer into
ordered **stages**, and each stage is a **node** holding a contiguous layer range.

```
gateway (x402/CIRC) → coordinator ──ACTIVATION──▶ stage A (layers 0:k)
   pod1: embed + lm_head + 0.5B draft + stage0       │
                       ▲                              ▼
                       └──────RESULT──────  stage B (layers k:n) …
   per token: embed → relay hidden state through each stage in order
              → norm + lm_head + sample → stream token
```

- **Wire**: framed, ChaCha20-encrypted, keyed. Message types already defined in
  `wire.py`: `HELLO` (*"worker→coordinator: identity + capabilities"*), `WELCOME`,
  `ACTIVATION`, `RESULT`, `KV_CTRL`, `PING/PONG`, `ERROR`, `BYE`. **`HELLO`/`WELCOME`
  are reserved but unused — the handshake was anticipated.**
- **State**: each stage keeps a **per-session KV cache** (`StageKV`), keyed by
  session id; `pos==0` begins a fresh sequence; `KV_CTRL` resets/rolls back.
- **Speed**: a 0.5B draft on the coordinator proposes K tokens; the split model
  verifies them per round-trip (predictive drafting).
- **Today's limits** (what this spec removes):
  - `CIRCUIT_STAGES` is a **static env list**, read once at startup; adding a node
    = edit + restart.
  - The coordinator **connects out** to each stage → nodes must be reachable inbound.
  - **One shared cluster key** → possession = membership, and every node in the
    chain sees the activations passing through it.
  - **Series circuit**: any stage down = whole service down (observed in prod).
  - Payment goes to one treasury; **nodes don't earn**.

---

## 1. Design goals (best practices)

1. **Dynamic topology** — add/remove nodes hot, no coordinator restart.
2. **Fault tolerance is first-class** — assume nodes are flaky and possibly hostile.
3. **Least privilege** — per-node credentials, not a shared master key.
4. **Idempotent + observable + gracefully draining** — safe retries, full metrics,
   clean exit.
5. **Clean security boundary** — permissioned now (allowlist), permissionless later
   (stake + verification) without a rewrite.
6. **Reuse** — the existing wire (`HELLO`/`WELCOME`), the shard-loader
   (`load_model_shard` already loads an arbitrary range), the node-client's ed25519
   identity, and the retired prototype's attribution math.

---

## 2. Architecture — add a control plane beside the data plane

Today there is only a data plane (activations over the wire). Add three pieces:

- **Control service** (logically part of the coordinator) — the source of truth for
  the live node set, layer assignments, and health; drives all topology changes.
  Exposed as an **HTTP/JSON control channel** (separate from the encrypted data
  wire) — easier to build, observe, and secure than overloading the binary wire.
- **Node agent** — what an operator runs: the **stage worker** (data plane) + a
  **control client** that registers, heartbeats, pulls its assigned layers, reports
  health, and tracks earnings. This is what `circuit-node-client` should become.
- **Relay / rendezvous** — a publicly reachable hop for NAT'd nodes (home GPUs).

Keep the **data wire** for `ACTIVATION/RESULT/KV_CTRL` + the `HELLO` handshake on
connect. Put **registration/assignment/heartbeat/drain** on the **control channel**.

---

## 3. Node lifecycle (the state machine)

```
DISCOVER → REGISTER → AUTHENTICATED → ASSIGNED → PROVISIONING → CONNECTED
        → READY → SERVING → (DRAINING | SUSPECT → DEAD) → LEFT
```

Join sequence:
```
node                          control service                coordinator/relay
 │  POST /register {id,caps,endpoint,wallet,sig} ──▶
 │  ◀── 200 {credential, assignment(layers k:k'), peers, relay?}
 │  pull weights for layers k:k' (shard download)
 │  open data-wire listener (or relay tunnel)
 │  HELLO {id, layers, credential} ─────────────────────▶ (coordinator dials/relays)
 │  ◀── WELCOME {ack, session-key}
 │  loop: heartbeat /hb {id, load, kv_sessions} every Ns ─▶
 │  serve ACTIVATION→RESULT on the data wire
 │  (leave) POST /drain → finish in-flight → BYE → LEFT
```

---

## 4. Registration & identity (Tier 1)

- **Identity** = an **ed25519 keypair** the operator's node generates (the
  node-client already has ed25519 identities). `nodeId = pubkey`.
- **Register** (`POST /register`), signed by the node key:
  `{ nodeId, gpu, vramFreeGB, maxLayers, endpoint|relayId, reachability, payoutWallet, ts, sig }`.
- **Admission**:
  - *Permissioned (now):* `nodeId` must be on an **allowlist** or present a valid
    **invite token**. Trust the operator.
  - *Permissionless (later):* open, but must post a **CIRC stake** (Sybil/slashing).
- **Per-node session key** — the control service issues each node its **own**
  short-lived data-wire key (derive from a master secret + nodeId, or wrap a random
  key in the response). **Replaces the single shared cluster key**, so revoking one
  node doesn't re-key the world, and a node only holds *its* key. Uses the reserved
  `HELLO`(caps) → `WELCOME`(assignment + key) handshake.

---

## 5. Dynamic layer assignment (Tier 1)

The control service maintains the **desired topology**: layers `[0..L)` partitioned
into ranges, each held by **R replicas** (see §6).

- **`Topology`** data model (replaces the static `_stage_addrs`):
  `ranges: List[Range{start,end, holders: Set[nodeId]}]` + `nodes: {nodeId → NodeInfo}`.
- **On join**, the assigner picks a range for the new node by priority:
  1. fill an **uncovered** gap, else
  2. raise an **under-replicated** range toward R, else
  3. **split** the largest range to rebalance load.
  Bin-pack by the node's `maxLayers`/VRAM. **Stable assignment** — minimize churn of
  existing holders.
- **Hot-apply**: update the routing table; **new** sessions use the new topology,
  **in-flight** sessions finish on the old. No restart. The node loads exactly its
  range via `load_model_shard` (already supports arbitrary ranges; AWQ-shard path is
  the one open eng item there).

---

## 6. Fault tolerance & re-routing (Tier 2 — *the gate*)

This is the heart of the spec: make the pipeline **not** a series circuit. You
cannot put a stranger's GPU in the chain without this — they *will* drop mid-token.

- **Replication factor R ≥ 2** — every layer range is held by ≥R nodes. Requests
  route to one holder; failures fail over to another.
- **Health detection** (defense in depth):
  - control-channel **heartbeats** (miss K → SUSPECT → DEAD),
  - data-wire **TCP keepalive + read timeout** (added in the wedge fix — already in
    `stage_worker.py`),
  - **request-level** failure (a relay/read error marks the holder SUSPECT).
- **Re-routing (per hop, per request).** The coordinator already holds the input
  hidden state for each hop (it relays them). So on a failed stage:
  `try holder_1 → on error, resend the same ACTIVATION to holder_2 of that range`.
  Bounded retries, exponential backoff, a **per-node circuit breaker**.
- **The KV-failover subtlety (correctness — don't miss this).** Each holder keeps
  *its own* per-session KV. If we fail over **mid-session** to a *different* holder,
  that holder has **no KV** for the session → its layers' attention would be wrong.
  Two options:
  - (a) **Warm replicas**: mirror every `ACTIVATION` to all R holders so each keeps
    live KV. Correct + instant failover, but ~R× the stage bandwidth/compute.
  - (b) **Re-prefill on failover**: when a session fails over, re-run the session's
    token history through the new holder for that stage to rebuild its KV, then
    continue. No steady-state cost; a one-time latency hit on the (rare) failover.
  **Recommendation: (b) for v1** — failures are rare, correctness is exact, cost is
  paid only on failover. Make it a tunable (warm replicas for premium SLAs later).
- **Re-balancing**: when a node dies and a range drops below R, the assigner
  schedules new holders (spares / new joiners) to restore R.
- **Graceful drain**: `POST /drain` → node stops taking new sessions, finishes
  in-flight, sends `BYE`, leaves; the assigner backfills its ranges.

This builds directly on tonight's hardening (per-connection threads + keepalive in
the stage worker) — that was the prerequisite; re-routing is the layer on top.

---

## 7. Reachability / NAT (Tier 4 — widen the pool)

The coordinator dials *out* today, so direct-connect works only for **public**
nodes (datacenter GPUs with a routable port). For **home GPUs behind NAT**:

- The node opens a **persistent outbound connection to a Circuit relay**; the
  coordinator reaches the node **through** the relay (which multiplexes data frames).
- The node advertises its `reachability` at registration (`public` | `relay`).
- The relay is a horizontally-scalable hop and a natural place for rate limiting and
  observation. Keep direct-connect for public nodes (lower latency).

---

## 8. Attribution & payout (Tier 3 — the incentive to join)

- **Record the route**: per request, the coordinator knows which node served which
  range for how many tokens. Log `{requestId, [{nodeId, layers, tokens}]}`.
- **Split**: the CIRC the gateway collected for the request is split **∝
  layers·tokens** across the serving nodes, minus a protocol/coordinator fee. (The
  retired prototype already has this attribution math — port it.)
- **Payout**: **accrue** per-node earnings in a ledger and **settle on-chain in
  batches** (periodic Token-2022 transfers to `payoutWallet`) — never per-request
  (gas would dwarf the payment). Best practice: an accrual ledger + a settlement job
  with a minimum-payout threshold.
- **Only pay for honest work** → ties to §9 (verification) for permissionless; under
  permissioned, the allowlist is the guarantee.

---

## 9. Trust & verification (Tier 5 — permissionless, the hard part)

Threats once nodes are untrusted: a node returns **garbage / a cheap approximation**,
or **sees the activations** flowing through it.

- **Honest-compute verification**:
  - **Redundant compute + compare** — R holders compute the same range; the
    coordinator compares a hash/sample of their outputs; a mismatch → eject + slash.
    Cheap, probabilistic, practical. *This is the realistic near-term answer.*
  - **Spot-checks** — re-run a random fraction of work on a trusted node.
  - **zkML proofs** — cryptographic proof of correct execution. Real but years out.
- **Privacy**: a node sees its layers' hidden states. Hard to fully prevent (secure
  enclaves / MPC are heavyweight). For permissioned: accept it. For sensitive
  workloads: restrict to trusted nodes.
- **Sybil / stake / slashing**: permissionless nodes **stake CIRC**; misbehavior
  (failed verification, downtime) **slashes** the stake. Identity is stake-backed.

Honest take: fully trustless inference is open research. The practical destination
is a **trust-minimized** network — allowlist or stake + **redundant-compute
comparison** + slashing — with zk as aspirational.

---

## 10. Observability & ops (best practices)

- Per-node: liveness, p50/p99 stage latency, tokens served, error rate, accrued
  earnings, KV sessions.
- Topology view: extend the existing `/v1/workers` (already reports layer ranges per
  machine) with holders, replication health, and churn.
- Alerts: any range **under-replicated**, elevated node churn, verification
  mismatches.
- Admin: drain/eject a node, pin an assignment, force re-balance.

---

## 11. Where it plugs into the existing code

| Component | Change |
|---|---|
| `engine/wire.py` | use the reserved `HELLO`(caps)/`WELCOME`(assignment+key) handshake; keep data frames as-is |
| `engine/coordinator.py` | replace static `_stage_addrs` with a live **`Topology`** (ranges → healthy holders); `_relay` routes per-hop **with failover + re-prefill**; host the control service; heartbeat tracking; re-balance scheduler |
| `engine/stage_worker.py` | add a **control client** (register, heartbeat, accept assignment, drain). Per-connection threading + keepalive already done |
| `engine/model.py` | `load_model_shard` already loads an arbitrary range (finish the AWQ-shard path) |
| **new** `engine/registry.py` | the control service: admission, assignment, health, re-balance, attribution ledger |
| **new** relay service | NAT traversal for home GPUs |
| **new** payout job | batched on-chain settlement from the accrual ledger |
| `circuit-node-client` | becomes the operator onboarding agent: run stage worker + control client + earnings UI |

---

## 12. Phased rollout (and why this order)

1. **Phase 1 — self-service join (permissioned, public nodes).** Control channel +
   registration + per-node identity + dynamic assignment + hot topology. Allowlisted,
   public-IP only. *Outcome: a vetted partner joins without us editing config.*
2. **Phase 2 — reliability (the gate).** Replication R≥2 + health + re-routing +
   re-prefill + re-balance + drain. **Must precede recruiting outside nodes** — this
   is what makes a flaky stranger safe to include.
3. **Phase 3 — earning.** Attribution + batched on-chain payout. *Now there's a
   reason to join.*
4. **Phase 4 — reach.** Relay/NAT traversal → home GPUs can join, widening the pool.
5. **Phase 5 — permissionless.** Stake + redundant-compute verification + slashing.
   The moonshot.

**Phases 1–3 = "a few vetted partners run GPUs and earn CIRC reliably"** — bounded
engineering (months), leveraging the reserved wire handshake, the retired prototype's
control plane, the shard-loader, and tonight's fault-tolerance groundwork. **Phases
4–5** widen and decentralize further; **5** is research.

**The single most important thing to build first is Phase 2 (fault tolerance)** —
the moment an outside node enters a series-circuit chain, reliability *is* the
product.

---

## 13. Holistic review — gaps & invariants the first draft missed

A second pass found these. Several are correctness-critical (get them wrong and
inference returns garbage or stalls), not nice-to-haves.

1. **Build order vs. priority.** "Build Phase 2 first" is about *importance*, but
   replication/failover **runs on top of** the Phase-1 foundation (the dynamic
   `Topology` + control service). Actual build order: **Phase-1 foundation →
   Phase-2 reliability**. You can't fail over to a replica you have no way to
   register and assign.

2. **Model/version consistency (correctness-critical).** Every node in a pipeline
   must run the **identical model — same weights, quantization, layer count,
   tokenizer**. A joiner on a different model/version produces hidden states the
   next stage misinterprets → silent garbage, no error. → The assignment carries a
   **model fingerprint** (e.g. HF revision + quant + dtype hash); a node that can't
   match it is rejected at registration.

3. **Complete-coverage invariant (the liveness condition).** At every instant, the
   set of *healthy* holders must cover `[coordinator_end, L)` **with no gap**.
   A single uncovered layer = generation is impossible. Re-balancing/failover must
   **never** leave a gap, even transiently; coverage is the property the whole mesh
   lives or dies on. The `Topology` must be able to answer `coverage_ok()` at all
   times and refuse to admit a routing that has a hole.

4. **Two different trust questions — don't conflate them.** *Did the node do the
   work?* is **coordinator-authoritative** — the coordinator routed the request, so
   it knows exactly who served which layers; attribution (§8) is un-gameable without
   any verification. *Did it do the work correctly?* is §9 (verification). Phase 3
   (earning) only needs the first; Phase 5 (permissionless) needs the second.

5. **Session affinity.** A generation session must **stick to the same holders**
   for its lifetime (their KV is warm). Failover (re-prefill, §6) is the *exception*
   path on a node death — not the normal one. Routing is per-session-sticky, not
   per-token-random.

6. **The coordinator is a special, central node — be honest about it.** It holds
   `embed` + `lm_head` + the draft + the orchestration loop. In the permissioned
   model it's **operator-run**; joiners take the middle/late *stages*. So this is
   "**decentralized compute, coordinated orchestration**," not a flat p2p network.
   Distributing/replicating the coordinator itself is a separate, harder problem
   (leader election / stateless coordinators behind the gateway) — out of scope for
   Phases 1–4, noted for later.

7. **Weight distribution + provisioning latency.** A joiner needs the weights for
   its assigned layers (~a stage's share = several GB to ~19 GB). Specify the
   **source** (HF mirror / a Circuit weight server / P2P) and note that a node is
   **not READY until its download completes** — provisioning is part of the
   lifecycle, not instant. The shard-loader lets a node fetch *only* its range.

8. **Churn damping (or the mesh thrashes).** Don't re-balance on every flap. Require
   a node to be **stably DEAD** (missed K heartbeats over a window) before
   reassigning its slot, and put a **cool-down** on re-balancing. Otherwise a
   flickering node triggers endless re-assign + re-prefill storms.

9. **The relay preserves end-to-end encryption.** The data wire is encrypted
   coordinator↔node; a relay (for NAT'd nodes) only **multiplexes ciphertext** — it
   never sees plaintext activations. The relay is itself a dependency for NAT nodes,
   so it must be **redundant**.

10. **Rolling upgrades.** Updating the model or engine across a heterogeneous mesh
    without a full outage: version-tag the topology, **drain-and-replace** a stage's
    holders one at a time, never mixing model versions within one pipeline.

11. **Chaos testing is the only way to trust fault tolerance.** Build a **local
    multi-node simulator** and **kill nodes mid-request** in CI. Failover code that
    isn't chaos-tested is failover code that doesn't work. (This is where we start
    below — the `Topology` is pure logic precisely so it's unit- and chaos-testable
    without GPUs.)

**Net:** the foundation to build first is the **`Topology` model + a control
service** (registration → assignment → health → coverage/rebalance), with the
**coverage invariant** and **model fingerprint** enforced from day one. Everything
else — failover, attribution, relay — hangs off that. Starting there.

---

## 14. Build status & operator runbook (what's shipped)

**Phases 1 & 2 are built, tested, and committed.** The control plane is library code
that the live coordinator mounts behind a flag — `CIRCUIT_MESH=1`. With the flag
unset the engine runs the original static path **byte-identically** (every change is
gated and regression-tested), so turning the mesh on is a deliberate, reversible op.

| Piece | Module | Test |
|---|---|---|
| Topology (slots, coverage invariant, holders, failover order, churn damping) | `engine/topology.py` | `test_topology` |
| Registry (admission + model-fp, per-node derived keys, session-affine `route_snapshot`, thread-safe, attribution ledger) | `engine/registry.py` | `test_registry` |
| Dynamic routing (session-pinned) + **mid-session failover + re-prefill** | `engine/coordinator.py` | `test_dynamic_relay`, **`test_failover`** |
| HTTP control channel (register/ready/heartbeat/drain, ed25519, reaper) | `engine/control_server.py` | `test_control_channel` |
| Stage-worker **control client** (join over the network) | `engine/stage_worker.py` | `test_node_join` |
| **Mounted on the live API**, gated | `engine/api.py` | `test_mesh_api` |

**Not yet built:** on-chain payout job (Phase 3 settlement — the ledger accrues, but
`settle()` isn't wired to Token-2022 transfers yet); NAT relay (Phase 4);
stake/redundant-compute verification (Phase 5); speculative-decode failover (greedy
paths have it). Re-prefill on failover is *reset-all-stages + replay the sequence* —
correct but O(sequence) on a death; fine while deaths are rare, optimize later.

### Enable the mesh on a coordinator

Set on the coordinator (alongside the usual `CIRCUIT_MODEL` / `CIRCUIT_KEY` /
`CIRCUIT_LOCAL_LAYERS`):

| Env | Meaning |
|---|---|
| `CIRCUIT_MESH=1` | turn the control plane on (default off) |
| `CIRCUIT_MESH_LAYERS` | total model layers (e.g. `64`) |
| `CIRCUIT_MESH_STAGES` | how many slots cover `[coordinator_end, LAYERS)` |
| `CIRCUIT_MESH_FP` | model fingerprint a joiner must match |
| `CIRCUIT_MESH_REPLICATION` | holders per slot (`R≥2` enables failover) |
| `CIRCUIT_MESH_ALLOWLIST` | comma-sep node-ids; empty = open (private net) |
| `CIRCUIT_MESH_SECRET` | hex master secret for per-node key derivation (defaults to `CIRCUIT_KEY`) |
| `CIRCUIT_CONTROL_PORT` / `_HOST` | control channel bind (default `18932`/`0.0.0.0`) |
| `CIRCUIT_MESH_VERIFY_SIG=1` | require ed25519 proof-of-key at registration |
| `CIRCUIT_MESH_DEAD_AFTER` / `CIRCUIT_REAP_INTERVAL` | churn damping (s) |

`coordinator_end` is taken from `CIRCUIT_LOCAL_LAYERS` (the co-located stage); the
coordinator co-locates `[0, coordinator_end)` and the mesh serves the rest. `/health`
and `/v1/workers` report the live topology when mesh mode is on.

### Join as a node

```
python3 -m engine.stage_worker \
  --control-url http://<coordinator>:18932 --node-id <ed25519-pubkey-hex> \
  --model <same as coordinator> --model-fp <CIRCUIT_MESH_FP> \
  --capacity-layers <N> --advertise-host <reachable-host> --device cuda
```

The node registers, receives a layer range + a derived per-node key, downloads/loads
those layers, serves, and heartbeats; it posts `/drain` on exit. No coordinator
config edit required — that's the whole point.
