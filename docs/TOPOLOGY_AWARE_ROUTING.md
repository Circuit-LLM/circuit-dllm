# Topology-Aware Routing — Design

**Status:** design / proposal
**Problem:** single-stream latency is unacceptable when nodes are geographically
distributed (the real, non-co-located case). We measured ~1.5 tok/s on a 72B mesh
whose coordinator sat in a different datacenter than its (co-located) workers.
Co-locating is **not** a product solution — contributed GPUs are scattered by
definition. This doc designs the genuine fix.

---

## 1. Root cause: the pipeline is a STAR

The coordinator drives the mesh from `coordinator._relay_dynamic`:

```python
for node in route:                       # route = one holder per slot, in LAYER order
    sock = self._conn_for(node)          # coordinator dials THIS node
    wire.write_frame(sock, ..., ACTIVATION, hidden)   # send hidden TO node
    mt, payload = wire.read_frame_keyed(sock, key)    # read result BACK to coordinator
    hidden = unpack(payload)
```

The activation returns to the coordinator **between every stage**. So for an N-slot
model the per-forward network cost is:

```
T_net  =  Σ_i  2 · RTT(coordinator, holder_i)            # N round-trips, all through the coordinator
```

This is the worst shape for distributed nodes: **every** layer-group hop pays the
coordinator↔node distance. Even though our 3 workers were on one machine (≈0 ms
between them), the star never used that adjacency — each of the 3 stages round-tripped
to the remote coordinator independently (≈3 × 2 × ~280 ms ≈ 1.7 s/round, matching the
measurement).

Speculative drafting (K tokens/round-trip) and request concurrency (throughput hides
latency across users — measured 3× aggregate at concurrency 4) both help, but neither
changes `T_net` for a single stream. The structure does.

---

## 2. The fix has three layers

### Layer A — Make the mesh latency-aware (foundation)

Today `topology.Node` knows `endpoint`, `capacity_layers`, `model_fp` — nothing about
*where* it is. Add:

- `region`/`geo`: derived from the node's public IP at `/register` (GeoIP, coarse —
  continent/metro is enough), plus an optional self-reported hint.
- A **latency matrix** `L[a][b]`: RTT between nodes and between the coordinator(s) and
  nodes. Bootstrap from geo distance; refine with cheap periodic RTT probes piggybacked
  on the existing heartbeat channel (nodes already heartbeat the control server).

This is pure metadata on `Topology`/`Registry`; no torch, unit-testable without GPUs.
It unblocks B and C.

### Layer B — Chain relay (the structural win)

Replace the star with a **chain**: the coordinator hands each holder the address of its
*next hop*; the activation flows forward node→node and only touches the coordinator at
entry and exit.

```
coordinator → holder(slot0) → holder(slot1) → … → holder(slotN) → coordinator
```

New per-forward cost:

```
T_net  =  RTT(coord, h0)  +  Σ_i RTT(h_i, h_{i+1})  +  RTT(h_N, coord)
```

The bulk of the hops become **direct node→node**, which Layer C then minimizes. Our
3-on-one-machine case would have collapsed to ~2 coordinator hops + 2 localhost hops
instead of 3 remote round-trips.

Costs/risks to handle in the design:
- **Failover** is harder than the star (where the coordinator just swaps a slot's
  replica). In a chain, a dead mid-chain node breaks the flow: the upstream node must
  re-point to a replacement next-hop. Mechanism: the coordinator owns the route; on a
  hop error it splices a replica into the chain and pushes the updated next-hop to the
  upstream holder, then the session re-prefills (the existing
  `_relay_with_failover` re-prefill path generalizes to this).
- **NAT traversal**: nodes must reach *each other*, not just the coordinator. Public
  nodes are fine; NAT'd nodes need the existing relay path (`reachability="relay"`) or
  get placed at chain ends (entry/exit) where only the coordinator dials them.
- Keep the star as a **fallback** when a chain can't be formed (all-NAT, tiny mesh).

### Layer C — Proximity ordering + nearest entry (the latency minimizer)

Layers must execute in fixed order (slot0→slot1→…), but **which node holds which slot**
is free, and with replication there are several holders per slot. So:

1. **Per-session route selection** (at `pos==0`, then pinned — affinity already exists
   via `_session_routes`): choose one holder per slot to **minimize the chain path
   length** `T_net` above. With one holder/slot ordered by layer, this is a shortest
   path through a layered graph — a clean DP/Viterbi over `L`. It degrades to today's
   behavior when there's no choice.
2. **Nearest entry**: route the request to the entry coordinator (or chain head)
   closest to the *user*, so the entry hop is short.
3. **Regional clustering (emergent):** when enough nodes exist in a region to cover the
   whole model, the minimizer naturally forms a region-local pipeline. Multiple regional
   pipelines = the throughput story (§3) with good single-stream latency each. No special
   case needed — it falls out of minimizing `T_net` with regional entry points.

---

## 3. Why throughput still matters (and is already working)

A decentralized GPU network's headline metric is **aggregate** tokens/sec across many
users, not one user's latency. Pipeline parallelism overlaps requests: while request A
waits on a hop, B/C/D compute at other stages. Measured: **1.61 tok/s single → 4.93
tok/s at concurrency 4** (≈3×, with `max_concurrency` only 4). Topology-aware routing
(§2) lifts the per-stream floor; concurrency lifts the ceiling. They compose.

---

## 4. Integration points (where the code changes)

| Change | File | Note |
|---|---|---|
| `Node.region`, GeoIP at register | `topology.py`, `control_server.py` | metadata only |
| latency matrix + RTT probes | `registry.py`, heartbeat path | piggyback on heartbeats |
| proximity route selection (DP) | `topology.py` `route()` / `registry.route_snapshot()` | replaces heartbeat-only ordering; pinned per session |
| chain relay (next-hop wiring) | `coordinator._relay_dynamic`, `stage_worker` serve loop | stage forwards to next; coordinator only entry/exit |
| chain failover splice | `coordinator._relay_with_failover` | generalize existing re-prefill |
| nearest-entry routing | API front / multi-coordinator | optional, for multi-region |

---

## 5. Phasing

- **P1 — foundation:** region/GeoIP + latency matrix + RTT probes. Pure metadata,
  unit-tested. No behavior change yet.
- **P2 — proximity route selection (star):** DP holder-per-slot picker minimizing
  `Σ 2·RTT(coord, h_i)`. Immediate win when replicas exist across regions; keeps the
  star (low risk). Validate with `tc netem`-injected latencies on a cheap mesh.
- **P3 — chain relay + proximity ordering:** the structural change to direct node→node
  hops with `Σ RTT(h_i, h_{i+1})` minimization; chain failover + NAT handling.
- **P4 — multi-region entry / nearest routing:** regional pipelines, route users to the
  nearest entry.

## 6. Testing without a global fleet

`tc netem` injects per-link delay/jitter on a small mesh (even all-local pods) to
emulate a geographic spread — more controllable and far cheaper than real cross-DC
placement. `topology.py` stays pure logic, so the route selector (P2) and chain ordering
(P3) get deterministic unit tests in `tests/test_topology.py` over synthetic latency
matrices, with `netem` runs for end-to-end confirmation.
```
