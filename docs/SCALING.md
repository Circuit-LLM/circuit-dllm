# Circuit DLLM — Scaling Architecture

How the DLLM grows from "my two pods serving one request at a time" to "a
join-able mesh serving x402-paid inference to anyone, at volume."

## Where we are today

```
gateway (x402) ── tunnel ── coordinator (pod1: embed/head/draft + stage0 0:31)
                                   │  encrypted TCP wire
                                   └─ stage worker (pod2: stage1 32:63)
```

- **One chain, two nodes.** The model is split by layer across independent,
  network-connected GPUs. Pod 2 is architecturally "someone else's GPU" — proof
  the mesh works across non-co-located machines.
- **Single-stream** — one request at a time (global lock in `api.py`), ~11 tok/s.
- **Static join** — adding a node = edit `CIRCUIT_STAGES`, hand-assign layers,
  restart the coordinator.
- **SPOFs everywhere** — any stage node, the coordinator, and the SSH tunnel each
  take the whole service down if they drop.

Capacity: ~100–140 reflect-sized calls/hour. Comfortably carries a swarm of
~500 agents doing sparse reasoning calls. Beyond that, or for external traffic,
the work below begins.

## The three independent scaling axes (do not conflate them)

| Axis | Solves | Technique |
|---|---|---|
| **Fit/split the model** | model too big for one GPU; bigger models | pipeline (have it) / tensor parallel |
| **Concurrency per chain** | many requests at once | **continuous batching** |
| **Total volume** | more than one chain can serve | **replica chains + router** |

Continuous batching ≠ multi-GPU. The mesh (pipeline) is the model split; batching
shares one chain among many callers; replicas multiply chains. They compose.

---

## What to build, in four planes

### 1. Control plane — self-service join + dynamic topology
Turns manual onboarding into "run a node and it joins."
- **Registry / discovery**: a node announces `{gpu, vram_free, layers_it_can_hold}`;
  the coordinator assigns it a layer range and folds it into the live topology
  **without a full restart** (hot add/remove of stages).
- **Reachability**: today the coordinator *connects out* to each stage, so a
  joiner must be reachable. RunPod gives a proxy port; a home GPU behind NAT needs
  a **rendezvous/relay or reverse tunnel**. This is the real "anyone" hurdle.
- **Heartbeat + health**: nodes heartbeat; a dead node is detected and its layers
  reassigned (see plane 3).

### 2. Data plane — concurrency + throughput
- **Continuous batching** in the engine: batch many requests' tokens through the
  pipeline together (the activations on the wire carry a batch). ~10–30× aggregate
  on the same GPUs. *Pragmatic version:* dense (padded) batch, drop speculative
  decoding on the batched path (speculative + batching is the hardest combo and
  batching wins on aggregate throughput anyway). HF layers already do batched
  attention — you build the scheduler + KV management, not attention from scratch.
- **Pipeline micro-batching** (cheaper interim, ~2×): keep both GPUs busy — pod 1
  computes request B's stage while pod 2 computes request A's. Reclaims the ~50%
  idle we measured. Tonight's per-connection stage threading is a step toward this.
- **Replica chains**: N independent full pipelines → linear ×N throughput.

### 3. Reliability plane — a chain is a series circuit
Any node down = whole chain down (we lived this). Random joiners make this the
central risk.
- **Redundancy**: ≥2 replica chains so one failure isn't an outage.
- **Re-routing**: detect a dead stage → reassign its layers to a hot spare or a
  node that replicates that range. Requires layer-range replication across nodes.
- **Coordinator HA**: the coordinator is a SPOF → multiple coordinators behind the
  gateway (stateless request routing) or leader election.
- **Kill the tunnel SPOF**: replace the single SSH tunnel with a redundant
  gateway→coordinator path.

### 4. Economic plane — x402 for anyone + operators earning
- **Entry gating**: built (gateway verifies CIRC payment per call).
- **Attribution + payout**: split each call's CIRC among the nodes that did the
  work (∝ layers/compute), so independent operators *earn*. The retired Node
  prototype had this concept; port it to the Python engine.
- **Pricing**: by tokens, not flat-per-call (deferred for now).
- **Trust/verification** (only if permissionless): prove a node ran its layers
  honestly — redundant compute / spot-checks / proofs. The hard research piece.
  A **permissioned** mesh (shared key, allowlisted operators) sidesteps it.

---

## Build sequence (pragmatic)

1. **Reliability first** — 2 replica chains + a router (health-checked, least-loaded,
   429 backpressure). Kills the SPOFs *and* doubles concurrency. Mostly ops + a
   router; minimal engine code. *This is the highest-value first move.*
2. **Concurrency** — continuous batching in the engine (the 10–30× multiplier).
   The big engine project; start with the pragmatic dense/no-speculative version.
3. **Control plane** — self-service join (registry + dynamic layer assignment +
   NAT relay) + operator attribution. This is what makes "anyone runs a GPU and
   joins + earns" real.
4. **Verification** — only if going fully permissionless (the moonshot).

## The strategic fork

- **Permissioned mesh** (you/partners run the nodes; architecture is join-able):
  build planes 1–3. Reliable, scalable, decentralized-capable. Months.
- **Permissionless network** (anyone joins + earns, trustlessly): add plane 4's
  verification. Open research; years.

Pick the target (agent count or external QPS) and the plane sequence follows.
