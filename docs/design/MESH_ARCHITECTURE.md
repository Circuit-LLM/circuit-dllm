# Circuit DLLM — Architecture v2: "Replicated GPU Mesh"

> Exploration doc. Compares the current internet-split pipeline with a
> "GPUs-compute / clients-route" mesh, and maps the full tradeoff space.
> Nothing here is built yet — this is the design map.

## 0. The one law that governs everything

**Per-token latency ≈ (number of separate machines a token passes through, in
sequence) × (network round-trip between them) + compute.**

A transformer is sequential (layer N needs N−1's output), so a token cannot skip
ahead. Every machine boundary the computation crosses costs one network
round-trip **per token**.

- **Today:** a single token crosses the internet ~6× (coordinator ↔ each client),
  each ~140 ms ⇒ ~2–3 s/token. The GPU (22 ms for 22 layers) and weak CPUs are
  *not* the bottleneck — the **machine boundaries in the per-token loop** are.
- **Goal:** drive the number of internet crossings in the per-token loop to **zero**.

There are two fundamentally different reasons to decentralize, and they have
opposite latency behavior:

| | Why | Latency |
|---|---|---|
| **Capacity pooling** (current) | Split a model so it runs on hardware no single node could hold | **Slow** — every shard boundary is a per-token hop |
| **Throughput replication** (proposed) | Each node holds the whole model; spread the *requests*, not the model | **Fast** — per-token loop stays on one machine |

The 7B is 4.7 GB; the L4 has 22 GB. **We are paying capacity-pooling's latency
cost for a capacity problem we don't have.**

---

## 1. Current architecture (v1 — internet-split pipeline)

```
                    ┌─────────── RunPod ───────────┐
  user → coordinator│ embed → GPU(L0-21) ──┐       │
        (tunnel)    │   ▲                  │        │
                    └───┼──────────────────┼────────┘
                        │ per token, 6× over the internet (~140ms each)
            VPS clients └→ c(L22) → c(L23) → … → c(L27) ─┘ → norm/lm_head/sample
```

- Coordinator: embed + output norm + lm_head + sampling + orchestration + x402 + API.
- 1 GPU worker (22 layers) + 6 CPU clients (1 layer each).
- Per-token hidden state ping-pongs coordinator ↔ workers ~7×.
- Payment: per-layer attribution.
- **Strength:** can run a model too big for any single contributor.
- **Weakness:** ~3 s/token, minutes to first token, fragile (one hop fails → request dies).

---

## 2. Proposed architecture (v2 — replicated GPU mesh)

Two planes:

### A. Compute plane — GPU inference nodes (replicas)
- Each GPU node runs a **complete** inference engine: tokenize/embed → all layers
  → lm_head → sampling. A standalone server (vLLM / llama.cpp / TGI).
- **A request runs entirely on ONE GPU node.** The per-token loop never leaves
  that machine → zero internet crossings per token.
- Nodes register: model(s) held, VRAM, current load, region/latency, price.
- **Model too big for one GPU?** A compute node can itself be a *co-located* GPU
  cluster (tensor/pipeline parallel over NVLink/PCIe/LAN — microsecond hops). From
  the network's view it's still "one logical compute node." Fast because local.

### B. Control / network plane — CPU node clients + registry
- CPU clients are the **decentralized front door**: accept user requests, verify
  x402 payment, pick a GPU node (by load/latency/price/region), proxy + stream the
  response back. Provide redundancy, edge caching, censorship-resistance.
- A **registry** (gossip/DHT, or a small set of coordinator nodes) tracks which
  GPU nodes are alive, what they hold, load, and price.
- Payment splits: GPU node (compute, majority) + CPU client (routing/service) +
  registry (discovery).

### Request flow (v2)
```
user → CPU client (verify x402) → ask registry for a GPU holding the model
     → forward request to chosen GPU node
     → GPU runs the FULL generation locally, streams tokens
     → CPU client relays the stream to the user
```

### Latency (v2)
- Per-token = GPU compute only (~20–50 ms on an L4 for 7B Q4; ~30–60 tok/s).
- Network = one request-in + one-way token streaming (NO per-token round-trips).
- **Result: ~30–50 tok/s (≈50–100× today), TTFT ~1–2 s (batched local prefill).**

---

## 3. Pros

1. **~50–100× faster per token, TTFT minutes → seconds.** The per-token loop is
   local. This is the difference between "demo" and "usable product."
2. **Throughput scales linearly with GPUs.** Each GPU serves whole requests; N
   GPUs ≈ N× concurrent capacity. Adding nodes makes the *network* stronger
   instead of making each request slower.
3. **Much simpler, more robust inference path.** No custom per-token tensor
   protocol, no layer-shard delivery/chunking, no coverage-gap/scramble bugs, no
   "one of 7 hops drops → token dies." Each GPU runs a standard engine.
4. **Graceful failure.** A GPU dying drops only its in-flight requests; the router
   reroutes. No single-pipeline fragility.
5. **CPU clients keep a real, valued role** — routing, payment, edge API,
   redundancy, registry/gossip. "Support the network" exactly as intended; they
   still earn (for service, not compute).
6. **Heterogeneous GPUs welcome.** Any GPU that fits the model can join — 3090,
   4090, L4, A100. No matched-hardware requirement.
7. **Stronger decentralization flavor.** Many independent *full-model* nodes
   (BitTorrent-seeder-like) is more censorship-resistant and resilient than one
   model split into fragile pieces.
8. **Distance stops mattering.** Route users to the nearest GPU; geography is no
   longer in the per-token loop.

---

## 4. Cons / risks / hard problems

1. **Model must fit on one GPU (or a co-located cluster).** Replication needs a
   single node to hold the whole model. 7B fits an L4; 70B (~40 GB Q4) needs a
   48 GB card or a co-located split; 405B needs a multi-GPU box. **This is the
   core tradeoff: you give up "run a model nobody's single box can hold" (v1's
   genuine niche, cake's whole pitch) for speed on models that *do* fit.**
2. **CPU nodes no longer do AI compute.** Identity shift: "everyone contributes
   *compute*" → "GPU owners compute; everyone else runs the *network*." Routing,
   payment, edge, gossip are real decentralized work — but if the brand is
   "pool everyone's CPU for AI," that changes.
3. **Verification / trust of GPU output (the big one).** When a single node does
   everything, how do you know it actually ran Qwen-7B honestly — not a cached
   answer, a cheaper smaller model, or garbage — and still claim payment? Options,
   all with costs:
   - Redundant execution (2+ GPUs, compare) — 2× cost.
   - Challenge/spot-checks — hard because sampling is stochastic.
   - TEE / attestation — hardware-dependent.
   - Reputation + staking/slashing — economic, not cryptographic.
   v1 doesn't fully solve this either, but v2 *concentrates* the trust in one node.
4. **Payment redesign.** From per-layer to per-request/token for GPUs + a service
   cut for routers + discovery for the registry. Tied to #3 (proving work to
   release payment).
5. **Scheduler / registry complexity.** Track GPU liveness, load, price, region;
   handle join/leave, hot-spotting, mid-stream failover. Decentralizing the
   registry itself (gossip/DHT, no single point) is more work.
6. **Per-GPU concurrency needs a batching engine.** To get high throughput per
   GPU, use continuous batching (vLLM excels; llama.cpp less so). Our custom
   `circuit-runner` is a per-token thing — v2 wants a real serving engine. That's
   a meaningful compute-engine swap.
7. **Cold start.** Each GPU loads ~5 GB into VRAM (seconds). Fine for long-lived
   nodes; worse for churny ones.
8. **GPU supply-side.** Needs GPU contributors; CPU-only folks can only run the
   network. Incentives must attract GPUs.
9. **Less dramatic demo.** The v1 "watch a token flow through layers across
   machines" viz is a great proof-of-concept. v2 (requests fan out to GPUs) is
   less visually striking — a marketing consideration.

---

## 5. The hybrid that's probably right

Don't pick one — let the **router choose the mode per model**:

| Model fits… | Mode | Speed |
|---|---|---|
| …on one contributor GPU | **Replicate** across many GPUs, load-balance | Fast, high throughput ✅ |
| …only on a co-located GPU cluster | Route to a **co-located split** node (fast local interconnect) | Fast |
| …nowhere but a pooled internet cluster | **v1 internet pipeline** (last resort) | Slow, but *possible* |

This keeps v1's unique "run the impossible" capability as a niche/fallback while
making the **common case (models that fit) fast.**

---

## 6. Current vs Proposed (at a glance)

| Dimension | v1 (internet pipeline) | v2 (GPU replicas + router) |
|---|---|---|
| Per-token | ~3 s | ~20–50 ms (50–100×) |
| Time-to-first-token | minutes | ~1–2 s |
| Adding nodes | single request gets **slower** | throughput goes **up** |
| Fault tolerance | one hop fails → request dies | GPU dies → reroute |
| Max model size | unbounded (pool capacity) | must fit 1 GPU / co-located cluster |
| CPU client role | compute a layer (bottleneck) | route / pay / edge (network) |
| Inference code | high custom complexity | standard engine + router |
| Trust/verification | distributed, unsolved | concentrated per node, unsolved |
| Decentralization flavor | capacity-pooling | throughput-replication |

---

## 7. Migration path (incremental, low-risk)

1. **Prove the thesis on the one L4 (no new hardware):** run the FULL 7B on the
   GPU node (it fits), make the coordinator a thin embed-free proxy. Measure — expect
   ~30–60 tok/s and ~1–2 s TTFT. This validates v2 with what we already have.
2. **Turn CPU clients into routers:** stop running layers; they become API edges
   that forward to the GPU node + handle x402. (They already proxy via the tunnel;
   generalize it.)
3. **Add a registry + load-balance** when a 2nd GPU joins. Test multi-GPU
   throughput (concurrent users).
4. **Verification + payment redesign** before admitting untrusted GPU providers
   (the gating hard problem).
5. **Add co-located-split mode** for bigger models; keep v1 pipeline as the
   "too-big-to-fit" fallback.

---

## 8. Decisions for the operator

1. **Identity:** "pool everyone's compute (incl. CPU)" or "fast decentralized
   inference service"? Determines whether the slow internet-split is core or niche.
2. **Trust model:** how much do you need *trustless* verification of GPU output
   (matters once untrusted GPUs join)? Reputation/staking vs redundant exec vs TEE.
3. **Target model sizes:** sets replicate vs co-located-split vs internet-pipeline mix.
4. **GPU supply:** can the incentive model attract GPU contributors?

---

## 9. Bottom line

The current design is slow for one structural reason: it splits a model that
fits on one GPU across 7 internet-distant machines, putting ~6 network crossings
in every token's path. The mesh/replication model removes those crossings
entirely for any model that fits on a node — yielding ~50–100× speedups and a
clean scaling story — at the cost of (a) needing GPUs that can hold the model,
(b) shifting CPU nodes from compute to network roles, and (c) confronting the
verification/trust problem head-on. The hybrid (replicate-when-fits,
co-located-split-when-big, internet-pipeline-as-last-resort) keeps the best of both.
