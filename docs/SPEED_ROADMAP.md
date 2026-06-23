# Speed Roadmap — making decentralized inference fast

**Status:** design / proposal
**Companion to:** `docs/TOPOLOGY_AWARE_ROUTING.md` (the routing foundation these build on).
**Constraint:** every technique here must preserve decentralization — no co-location, no
assuming nodes are nearby. They make *genuinely distributed* nodes fast.

---

## The frame

To emit one token, the activation must walk through every layer **in order**, and the
layers live on distant machines. So a token costs a **serial chain of WAN hops**. Single-
stream latency ≈ `Σ (hop RTTs)`. Every idea below attacks that sum on one of four fronts:

1. **Fewer hops per token** — skip layers, pack fatter stages, replicate regionally.
2. **More tokens per hop-sweep** — speculation trees, lookahead.
3. **Cheaper / hidden hops** — replica hedging, async pipelining, concurrency.
4. **No hop at all** — prefix-KV and response caching.

They compose because they attack *different* fronts (see §Compose). They are NOT all free —
each has a tradeoff called out explicitly.

---

## Front 1 — Fewer hops per token

### 1.1 Early-exit / adaptive depth  ⭐ (biggest avg-case win)
**What:** for "easy" tokens (obvious next word), stop after the first M of N layer groups
instead of all N.
**Why distributed:** on one GPU this saves a little compute; here **skipping a layer group
skips a whole network round-trip.** Most tokens are easy, so average hops drop sharply.
**How:** each stage carries a tiny **exit head** (a learned linear → logits, or a confidence
probe on the hidden norm/entropy). At a chain node K, if confidence ≥ τ, short-circuit:
send the hidden straight back to the coordinator for `norm`+`lm_head`, skipping nodes K+1…N.
The draft (§2.1) can even *predict* the exit depth so the path is chosen up front.
**Tradeoff:** quality — exiting too early degrades output. Needs calibrated thresholds
(per-position confidence) and a quality gate. **Heaviest lift** (needs exit heads +
calibration, possibly light training). Highest payoff.
**Depends on:** chain relay (to "skip" downstream nodes cleanly).

### 1.2 Fewest-fattest stages  ⭐ (structural, cheap)
**What:** pack as many layers per node as VRAM allows → minimize stage count → fewer hops.
A 24 GB card holds ~40 layers of a 4-bit 72B, so run it as **2 fat stages, not 4**.
**Why distributed:** half the stages = half the inter-node hops, directly.
**How:** the topology planner (`topology.py` slot construction) chooses `num_stages =
ceil(total_layers / max_layers_per_node)` from joined nodes' `capacity_layers`, preferring
the fewest stages that cover the model, then assigns by proximity (`TOPOLOGY_AWARE_ROUTING`).
**Tradeoff:** raises the bar to contribute (favors beefier GPUs); fewer stages = less
parallelism headroom for very long models. Pure win when capable nodes exist.
**Depends on:** topology-aware assignment (P1/P2).

### 1.3 Regional pipeline replication (emergent)
**What:** when node density allows, a full pipeline replicates per region; users hit the
nearest complete pipeline.
**Why distributed:** local-latency per user without co-locating anything — independent
contributors who happen to be regional. CDN-for-inference.
**How:** falls out of proximity routing + nearest-entry once a region has enough nodes to
cover all slots. No special mechanism beyond §TOPOLOGY_AWARE_ROUTING + multi-entry.
**Tradeoff:** needs enough nodes per region to cover the whole model; emerges with scale.

---

## Front 2 — More tokens per hop-sweep

### 2.1 Latency-scaled speculation trees  ⭐
**What:** instead of K *linear* draft tokens per round-trip, send a **tree/forest** of
candidate continuations and verify all branches in one sweep. (Groundwork:
`SPECULATION_FOREST.md`, `specdecode.py`.)
**Why distributed:** you pay for the round-trip regardless — so verify as much as possible
in it. **The worse the RTT, the bigger the tree should be.** Tree width/depth becomes a
function of the *measured* RTT of the assigned pipeline.
**How:** extend `speculative_greedy*` to propose a token tree from the draft; pack the tree
into one `BATCH_ACTIVATION` sweep (reuse the batched-wire path, `wire.BATCH_ACTIVATION`);
verify with a tree attention mask; accept the longest matching branch. Tree size = f(RTT)
from the latency matrix.
**Tradeoff:** wasted draft compute on rejected branches; benefit shrinks at high batch /
throughput (speculation always does). Coordinator-side draft compute grows with tree size.
**Depends on:** latency matrix (P1) to size the tree; batched-wire path (exists).

### 2.2 Lookahead / Jacobi decoding
**What:** draft-free parallel token generation (n-gram guesses refined by Jacobi iteration).
**Why distributed:** more compute per round-trip, fewer rounds; no separate draft model.
**How:** a parallel-decode loop at the coordinator; verify n-grams through the pipeline.
**Tradeoff:** less mature than tree-spec for chat; interacts with KV layout. A "maybe."

---

## Front 3 — Cheaper / hidden hops

### 3.1 Replica hedging — race the replicas  ⭐ (cheap, immediate)
**What:** send each hop's activation to **2+ holders of that slot simultaneously**, use the
first response, cancel the rest.
**Why distributed:** we already replicate slots for *failover* — reuse that redundancy for
**speed**. Every hop silently routes around the slowest/most-distant replica; kills tail
latency from one bad node on the path.
**How:** in `coordinator._relay_dynamic`, for a slot with ≥2 routable holders, fan out the
`ACTIVATION` frame to the top-2 (by latency matrix), take the first `RESULT`, drop the
other conn. Falls back to single-send when only one holder.
**Tradeoff:** **costs throughput** — duplicate compute on the loser. A latency↔throughput
knob (hedge under low load, single-send under high load). Decentralized + low-complexity.
**Depends on:** replication active; latency matrix (P1) to pick the 2 best (works blind too).

### 3.2 Async speculative pipelining
**What:** the draft runs *ahead* continuously so verification of chunk N overlaps drafting
chunk N+1 — the expensive pipeline is never idle waiting on the local draft.
**How:** decouple draft and verify into a producer/consumer; the `async-pipelined
speculative decoding` seam is already noted in `api.py`.
**Tradeoff:** complexity; wasted work when a verification invalidates queued drafts.

### 3.3 Concurrency / continuous batching (proven)
**What:** pipeline many requests so WAN latency hides in aggregate throughput.
**Status:** measured **1.61 → 4.93 tok/s at concurrency 4** (`CIRCUIT_MAX_CONCURRENCY`,
Win A live). Continuous batching (Win B, `CIRCUIT_BATCH`) is built + gated. Push both.
**Tradeoff:** helps aggregate, not single-stream.

---

## Front 4 — No hop at all

### 4.1 Prefix KV caching  ⭐ (huge for chat)
**What:** cache the KV of shared prefixes (system prompt, common preambles) at each node so
the prefill **never re-crosses the network** for a repeated prefix.
**Why distributed:** prefill is a full network traversal of the prompt; for a service with a
fixed system prompt, the first request pays it and the rest don't.
**How:** content-hash the prefix; each stage keeps a small LRU of `prefix_hash → KV slice`
for its layers; coordinator sends `prefix_hash`, nodes load cached KV and only process the
new suffix. Builds on `StageKV` / the per-session cache.
**Tradeoff:** node memory for the cache; invalidation on prefix change. Mostly pure win.

### 4.2 Semantic / response caching (edge)
**What:** popular or near-duplicate queries answered from an edge cache — zero traversal.
**How:** exact + embedding-similarity cache at the entry coordinator.
**Tradeoff:** correctness/freshness for dynamic queries; opt-in per route.

---

## Moonshots (anything-goes frontier; exactness-trading)

- **Speculative activation prediction** — a tiny per-node model guesses the *incoming*
  activation so a node starts before it arrives, then corrects. Likely too lossy; novel.
- **Stale/async-activation decoding** — tolerate slightly-stale activations (async-SGD
  style) to break the sync barrier. Trades exactness for throughput.

---

## The synthesis: a latency-adaptive decoder

The standouts unify into one idea: **each request measures the RTT of its assigned pipeline
and adapts** — big trees (2.1) + aggressive early-exit (1.1) + replica hedging (3.1) on
high-latency chains; lean linear decode on fast ones. The decoder reacts to network weather.
Novel for decentralized inference; sits directly on the topology-aware routing foundation.

---

## Do they compose? (answering "would all 5 help everything?")

They attack different fronts, so they **stack multiplicatively** — but it is NOT a uniform
"everything gets better"; each carries a tradeoff and two interact.

| Technique | Front | Single-stream latency | Aggregate throughput | Quality | Cost |
|---|---|---|---|---|---|
| Early-exit (1.1) | fewer hops | ⬆⬆ avg | ⬆ (less compute) | ⬇ risk (tune τ) | exit heads + calibration |
| Fewest-fattest (1.2) | fewer hops | ⬆⬆ | ⬆ | – | needs beefier nodes |
| Spec trees (2.1) | more/hop | ⬆⬆ | ⬇ at high batch | – (exact) | draft compute |
| Replica hedging (3.1) | cheaper hop | ⬆⬆ tail | **⬇ (dup compute)** | – | extra compute |
| Prefix KV cache (4.1) | no hop | ⬆⬆ TTFT | ⬆ | – | node memory |

Interactions: **early-exit × spec-trees** is the one non-trivial combo (branches that exit
at different depths need care in the tree verifier). The rest are orthogonal. Net: all five
compound the speedup, but you're tuning a set of latency↔{quality,throughput,memory} knobs,
not flipping a single "fast" switch.

---

## Sequencing — when to do what

**Trigger:** this work pays off when the **decentralized mesh becomes the product direction**
(today prod is the 32B; the mesh is proven-but-not-productized). Single-stream latency is the
gate to productizing the mesh at all — so the cheap wins double as the "is the mesh viable?"
experiment. Build **incrementally and measured**, not all-five-blind: the cheap, low-risk
three first, measure on a `tc netem` testbed, *then* commit to the heavy two.

- **P0 (prereq):** topology-aware foundation — latency matrix + region metadata
  (`TOPOLOGY_AWARE_ROUTING` P1). Everything latency-adaptive needs it.
- **Wave 1 — cheap, low-risk, big lift (do together):**
  - **Replica hedging (3.1)** — tail latency, ~free, decentralized.
  - **Fewest-fattest stages (1.2)** — structural hop reduction, lives in the topology planner.
  - **Prefix KV cache (4.1)** — TTFT, high value for real chat, independent.
  - Measure each on netem (injected RTT/jitter) + the existing unit-test harness.
- **Wave 2 — single-stream depth:**
  - **Latency-scaled spec trees (2.1)** — builds on the spec-forest groundwork + the matrix.
  - Async speculative pipelining (3.2) if Wave 1 shows the draft is on the critical path.
- **Wave 3 — the heavy hitter, deliberate:**
  - **Early-exit / adaptive depth (1.1)** — biggest avg-case win, biggest lift, quality-
    sensitive. Start as a parallel research/calibration track; productionize behind a
    quality gate.
- **Emergent (no build):** regional replication (1.3) appears with node density.

## Testing

`tc netem` injects per-link delay/jitter on a cheap throwaway mesh to emulate a geographic
spread (more controllable than real cross-DC placement). `topology.py`/`specdecode.py` stay
pure logic → deterministic unit tests over synthetic latency matrices and token trees, with
netem runs for end-to-end confirmation. Gate every technique behind an env flag (like
`CIRCUIT_BATCH`/`CIRCUIT_MAX_CONCURRENCY`) so the default path stays byte-identical until proven.
