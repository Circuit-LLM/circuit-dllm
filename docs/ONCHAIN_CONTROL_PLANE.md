# On-Chain Control Plane — a leaderless control plane anchored on Solana

**Status:** SPEC (not built). The design we build against; do it right, not fast. Companion to
`docs/FLOATING_COORDINATOR.md` (removed the head funnel; left the control plane as the last singular
piece) and `docs/CUTOVER_FLOATING.md` (the control plane is currently one pod = a SPOF).
**Constraints (non-negotiable):** no central host / no VPS dependency; must hold on a *scattered,
high-RTT, churning* mesh; the control plane is **off the per-token path** (hit once at session start
+ on failover), so it can afford slow coordination that would be fatal on the token path; lean into
the Solana foundation the project already has (ed25519 node identities, StakePoint, CIRC payouts).

---

## 1. Problem

The floating coordinator removed the single *head/orchestrator* funnel, but the **control plane**
(registry/router: membership, slot layout, trust/bans, routing) is still a single process on a single
pod. If it dies, new routing stalls until it restarts. Moving it to the VPS just trades a GPU-SPOF for
a host-dependency — and the operator explicitly does not want a VPS. We want it **distributed, on the
nodes, with no single point of trust or failure.**

## 2. The key insight — split the state by change-rate, not by node

The control plane's state is two very different things:

| State | Examples | Rate | Needs | Home |
|---|---|---|---|---|
| **Authoritative** | model fp, slot layout, replication target, node identity, **stake**, **trust/bans** | slow (joins, audits) | strong consistency + verifiability | **on-chain (Solana)** |
| **Soft** | which nodes are live *now*, per-holder load, RTT, which holder serves a slot, a session's route | fast (heartbeats, per request) | availability, eventual consistency | **in-mesh (leaderless)** |

The chain is the consistency + trust layer for the slow, security-critical state (which it is *good*
at, and which *benefits* from being publicly verifiable + permissionless). The mesh handles the fast
routing with **no leader and no quorum**. The result is not "the control plane moved" — it's **there
is no control plane process anymore**; the chain anchors it and every node derives the rest locally.

## 3. On-chain layer (Solana) — the authoritative anchor

A small Anchor program owns:
- **`MeshConfig` (one PDA):** `model_fp` (`qwen2.5-72b-awq`), `num_layers` (80), the **slot layout**
  (boundaries, e.g. `[0,40),[40,80)`), `replication` target, `version`. Operator/governance-writable;
  changes rarely. This is the "topology contract" every node agrees on.
- **`Node` (PDA per node_id = ed25519 pubkey):** `role` (orchestrator | holder), `trust`
  (probation | trusted), `banned`, `payout_wallet`, `joined_at`, and a **stake reference** (read from
  StakePoint — do NOT duplicate stake on-chain). A node creates/updates its own PDA on join (signed by
  its key → trustless identity). `trust`/`banned` are written only by the **auditor authority**
  (the existing `docs/VERIFICATION.md` probation→trusted/evict flow, now writing on-chain).

**NOT on-chain** (deliberately): a node's *endpoint* (host:port — changes on every pod/proxy restart)
and its *liveness* (heartbeats) — both too frequent. Those are learned in-mesh (§4). So on-chain
**write volume is low**: a one-time node-join tx, rare trust/ban txns, rare layout edits. Cost +
finality latency are irrelevant off the hot path.

**Reads:** orchestrators + the gateway read `MeshConfig` + the `Node` set via RPC, **cache** them, and
**subscribe** to account-change websockets for push updates. A node's stake/trust is now *publicly
verifiable* — anyone can audit who's in the mesh, which is the actual decentralization win over a
private registry.

## 4. In-mesh layer — leaderless, deterministic routing

Every orchestrator is a self-sufficient router; there is no shared routing state to replicate.

- **Liveness:** holders heartbeat to the orchestrators (small, frequent); each orchestrator tracks the
  live set locally. A missed-beat window drops a holder from *that* orchestrator's view (→ failover).
- **Slot self-assignment:** a holder reads the on-chain layout, picks the **least-covered slot** from
  its gossiped view, loads it, and announces `serves [a,b)` in its heartbeat. Collisions (two holders,
  same slot, while another is uncovered) resolve by a deterministic tie-break (higher node_id drains +
  re-picks) → coverage converges in a few heartbeat rounds. No central `_pick_slot`.
- **Routing (`acquire_route`) → rendezvous hashing (HRW):** for each slot, pick the live holder that
  maximizes `hash(session_id, holder_id)`. **Any orchestrator independently computes the SAME route
  for a session** given the same live set — so a re-home onto a *different* orchestrator lands on the
  **same holders' warm KV with zero shared pin state** (this replaces the §6 session-affinity pin of
  the floating-coordinator doc with a stateless, leaderless equivalent). A dead replica simply drops
  out of the max → the next-best holder is chosen → only that slice re-prefills.
- **Entry selection (`acquire_entry`) moves to the gateway:** the gateway reads the on-chain
  orchestrator set, health-checks them, and HRW/round-robins. No central entry service.
- **Load balancing:** HRW's natural spread + an inverse-load weight each orchestrator estimates
  locally; approximate is fine (it's balancing, not correctness).

## 5. Request flow

```
gateway: read on-chain orchestrators (cached) → health-filter → HRW pick orchestrator O
client → O
O: read MeshConfig (cached) + live-holder map (heartbeats); HRW-route session over slot replicas
O: embed → relay tree through the chosen holders (warm KV by session-id) → norm/lm_head → stream
  (re-home: any survivor O' recomputes the identical HRW route → re-attaches, no pin, no re-prefill)
```

The chain is touched **zero times per request** — only its cached projection is. Writes hit the chain
only on join / trust change / layout edit.

## 6. Failure modes (no single death is a global outage)

- **Orchestrator dies:** the gateway health-check drops it; HRW picks another. No state lost.
- **Holder dies:** orchestrators see heartbeats stop → drop it → HRW routes its slot to another live
  replica (needs replication ≥ 2); only that slice re-prefills.
- **Solana RPC unreachable:** nodes serve from the **cached** on-chain view (membership/layout don't
  change fast). Routing continues (in-mesh liveness + HRW). Only *new joins / trust changes* pause
  until RPC returns. Use ≥2 RPC endpoints (e.g. Helius + a fallback) to make this rare.
- **A node lies** (claims a slot it can't serve, fakes liveness): the existing trust/audit flow keeps
  it probationary until verified; HRW only routes to trusted+ready holders; bans are on-chain.
- **Split-brain:** impossible for authoritative state (the chain is the single source of truth). Soft
  liveness views may briefly differ across orchestrators during churn, but that only causes a failed
  hop → failover, never incorrect output.

## 7. Mapping to existing code (the seams exist)

- `engine/registry.py` `Registry` splits into **`OnChainRegistry`** (reads `MeshConfig` + `Node` PDAs,
  caches, subscribes) + **`MeshLiveness`** (heartbeat-tracked live set, in-process). The persisted
  `CIRCUIT_REGISTRY_STATE` is replaced by the chain as the durable store.
- `acquire_route` → an **HRW router** over `MeshLiveness` (deterministic; the session-affinity pin
  from FLOATING_COORDINATOR §6 becomes the HRW property — *delete* the pin, keep the behavior).
- `acquire_entry` → moves into the gateway (`inference-gateway.js`): read on-chain orchestrators +
  health + HRW. The gateway's `resolveTarget` already abstracts "where do I send this" — point it at
  the chain instead of a control-plane RPC.
- ed25519 node identity (already used to sign `/register`), StakePoint (`verifyStake`), the trust/audit
  promote/evict flow, and payout wallets are **already** the inputs — this spec moves their *record*
  on-chain rather than inventing anything.
- The control-plane `/register`, `/route/*`, `/entry/*` RPCs become **unneeded** (registration = a
  node-PDA tx; routing = local HRW). Keep them during migration (§8) as a compatibility shim.

## 8. Build order (do it right; byte-identical gates; current control plane keeps working)

1. **Anchor program + accounts.** `MeshConfig` + `Node` PDAs + write-authority rules (node self-write
   identity; auditor-only trust/ban). Devnet first. DoD: unit-tested program; a node can create its
   PDA + the auditor can flip trust.
2. **Mirror + read-validate (shadow).** A writer mirrors the *current* control-plane registry on-chain.
   Orchestrators read the chain (cached) **alongside** the live control plane and assert their derived
   slot layout + node set + **HRW route == the current `acquire_route`** for the same session/live-set.
   DoD: byte-identical routing decisions over a sweep; no behavior change (read-only).
3. **Route from the chain + HRW (leaderless), gateway entry from the chain.** Orchestrators stop calling
   the control plane and route from `OnChainRegistry` + `MeshLiveness` + HRW; the gateway picks
   orchestrators from the chain. The central control plane still runs as a fallback. DoD: parity +
   coverage + failover + **re-home affinity (HRW lands the same holder, no pin)** on a scattered mesh;
   orchestrator-kill + control-plane-kill both continue serving.
4. **Retire the central control plane.** Fully leaderless. DoD: kill *every* control-plane process and
   the mesh still registers (PDA txns), routes, re-homes, and serves.
5. **(Optional) governance** for `MeshConfig`/layout edits (multisig/DAO) — last, off the hot path.

## 9. Risks / open questions

- **The Anchor program is real on-chain dev** (+ an audit before it owns trust/bans). Bootstrap option:
  a single operator-owned config account + StakePoint for stake + an off-chain-signed node list anchored
  by hash, before the full program — gets decentralization incrementally.
- **RPC dependency.** Mitigate with ≥2 providers + aggressive caching + ws subscriptions; the cached
  view keeps routing alive through an RPC outage.
- **Slot self-assignment convergence** under heavy churn — needs the collision/back-off rule tuned;
  for the operator's stable fleet it can pin slots in `MeshConfig` initially.
- **HRW vs load** — HRW gives affinity + spread but not perfect load-balance; add inverse-load weights;
  measure on a scattered mesh.
- **Endpoint discovery** is in-mesh (heartbeat) — a brand-new orchestrator must learn holder endpoints
  from heartbeats/gossip before it can route; bootstrap from a seed-node list (already in config).

## 10. One-line summary

Put the **slow, security-critical, verifiable** state (layout, identity, stake, trust, bans) **on
Solana**, derive the **fast** state (liveness, slot assignment, routes) **locally on every node via
heartbeats + rendezvous hashing** — so there is no control-plane process, no quorum, no VPS, and no
single point of failure or trust: the chain anchors the mesh, and any node can route any session.
