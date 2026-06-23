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

### 2.2 EAGLE-3 drafting  ⭐⭐ (highest-ROI single lever — more accepted tokens per round-trip)
**What:** replace the vanilla 0.5B draft with an **EAGLE-3 head** — a 1-layer draft (~1-3B)
that autoregresses at the *feature/hidden-state* level, fusing activations from every target
layer through a learned gate and reusing the target's own LM head. State-of-the-art draft
acceptance.
**Why distributed (this is the whole game):** in a high-latency mesh the network round-trip
is the fixed cost, so **acceptance per round-trip is everything**, and the draft runs on the
coordinator → **zero added hops**. Our live mesh measured **acceptance 0.389 / 2.56 tokens
per round** with the untrained 0.5B; EAGLE-3 reports **0.80-0.88** acceptance on the Qwen
family → roughly **2× the tokens per round-trip → ~half the round-trips per token.** That is
a near-direct ~2× on single-stream latency, stacking on top of chain relay and trees.
**How:** drop an EAGLE-3 head onto the coordinator (it already holds embed/lm_head + early
layers); feed it the target's fused features; it proposes the tree that §2.1 verifies in one
sweep. `specdecode.GreedyDraft` → `EagleDraft` behind `CIRCUIT_DRAFT_KIND=eagle`.
**Availability (researched 2026-06):** pre-trained EAGLE-3 heads exist for **Qwen2.5-72B**
(e.g. AngelSlim / the EAGLE repo) → mostly download+integrate; training a custom head is
~2-4h on 4×H100 with the target's activations as supervision. Tokenizer-shared with the 72B.
**Tradeoff:** the head is model-specific (one per target/rev); ~1-3 GB extra on the
coordinator; integration must match the target's hidden-state interface. Pure win otherwise.
**Depends on:** the speculative path (exists); pairs with §2.1 trees (EAGLE-3 is tree-native).

### 2.3 Speculation swarm  🧪 (the per-node-0.5B idea, done right — uses idle decode compute)
**What:** decode is **memory-bandwidth-bound**, so each node's compute units sit ~idle during
the big model's forward. Put that free capacity to work: each node runs a tiny local draft
that proposes a *diverse* candidate continuation; the coordinator **harvests these during the
verification sweep that's already in flight** (a side-channel on connections already open — NO
extra round-trip) and assembles a wider tree. More diverse branches → more accepted tokens per
expensive round-trip, at ~zero marginal cost.
**Why distributed:** turns the contributed GPUs from "just pipeline stages" into a speculation
engine, monetizing compute that's otherwise wasted on a memory-bound decode.
**Why it's 🧪 not ⭐:** the win hinges on harvesting drafts *without adding a hop* — the timing
(draft round N+1 during verify round N) and the diversity-vs-redundancy of independent drafts
are the open questions. EAGLE-3 (§2.2) captures most of the "more tokens/round" benefit with a
single strong coordinator-local draft and no coordination risk — so **EAGLE first, swarm as the
research bet on top.** A swarm of *EAGLE* heads (one per node) is the ambitious endgame.
**Depends on:** §2.1 trees + §2.2 EAGLE as the per-node drafter; async-pipelined harvest (§3.2).

### 2.4 Lookahead / Jacobi decoding
**What:** draft-free parallel token generation (n-gram guesses refined by Jacobi iteration).
**Why distributed:** more compute per round-trip, fewer rounds; no separate draft model.
**How:** a parallel-decode loop at the coordinator; verify n-grams through the pipeline.
**Tradeoff:** less mature than tree-spec for chat; interacts with KV layout. Superseded by
EAGLE-3 for our case — keep as a fallback if an EAGLE head proves hard to obtain/train.

---

## Front 3 — Cheaper / hidden hops

### 3.1 Replica hedging — race the replicas  (DOWNGRADED — KV affinity kills the naive version)
**What (original idea):** send each hop's activation to 2+ holders of that slot, use the
first response, cancel the rest — route around the slowest replica every hop.
**Why it does NOT work per-token here (the finding):** this is an **autoregressive KV
pipeline** — each holder caches the session's KV for *its* layers, and token N's forward
needs the KV from tokens 0…N-1 *on that same node*. So a slot's holder is **pinned for the
session** (`coordinator._session_routes`, "affinity → warm KV"). You can't send token N to
replica A and token N+1 to replica B: B never saw 0…N, so its KV is empty → wrong output or
a full re-prefill. Per-token hedging therefore requires **mirroring every token to both
replicas** (keep both KVs warm) = ~2× compute *always*, not just under contention.
**What survives:** **race-to-pin** — race the *first* token across the top-2 replicas and
pin the winner for the session. But that doubles the *prefill* (the loser's prefill is
wasted), and **proximity routing (P1, already done) already picks the fastest replica at
pin time from the RTT prober with ZERO per-request cost.** So race-to-pin adds cost for a
marginal gain over what we already have.
**Conclusion:** proximity routing subsumes the practical benefit; true per-token hedging is
high-cost / low-ROI for a KV pipeline. **Not building it.** (Failover across replicas — the
*reliability* use of replication — stays as-is.)
**Lesson:** KV affinity is the constraint that makes distributed *decode* fundamentally
different from stateless request hedging. It shapes the whole roadmap (see Early-exit, Chain).

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
| **Chain relay (P3)** | fewer/cheaper hops | ⬆⬆ structural | ⬆ | – (exact) | wire + failover rework |
| **EAGLE-3 draft (2.2)** | more/hop | ⬆⬆ (~2× tokens/round) | ⬇ at high batch | – (exact) | model-specific head, ~1-3 GB |
| Early-exit (1.1) | fewer hops | ⬆⬆ avg | ⬆ (less compute) | ⬇ risk (tune τ) | exit heads + calibration |
| Fewest-fattest (1.2) | fewer hops | ⬆⬆ | ⬆ | – | needs beefier nodes |
| Spec trees (2.1) | more/hop | ⬆⬆ | ⬇ at high batch | – (exact) | draft compute |
| Proximity routing (P1 ✓) | cheaper hop | ⬆ | – | – | ~free (RTT prober) |
| Prefix KV cache (4.1) | no hop | ⬆⬆ TTFT | ⬆ | – | node memory |
| Speculation swarm (2.3) | more/hop | ⬆ (research) | – (idle compute) | – (exact) | harvest-timing risk |

Spec trees (2.1), EAGLE-3 (2.2) and speculation swarm (2.3) all live on the same axis
(tokens-per-round) and **compose**: EAGLE is the strong per-draft model, trees verify many of
its branches per sweep, the swarm widens the tree with idle per-node compute. Build them in
that order.

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

- **P1 (DONE ✓) — latency-aware foundation + proximity routing:** `Node.region` +
  coordinator region + RTT map (measured > region-estimate > default), active TCP-connect
  RTT prober, holder ordering by RTT (`CIRCUIT_ROUTE_LATENCY`, default off = byte-identical).
  Unit-tested (`test_topology_latency`, `test_rtt_probe`). This already **picks the fastest
  replica per slot at pin time** — i.e. subsumes the practical benefit of replica hedging
  (which KV affinity rules out, see §3.1).
- **Wave A — the two biggest levers, built together, measured on ONE 72B mesh bring-up
  (current focus):**
  - **Chain relay (P3 / docs/CHAIN_RELAY.md)** — the structural fix: stages forward node→node,
    coordinator only at entry/exit. Turns N coordinator round-trips into one forward sweep.
    Pure-logic core (route→chain, next-hop, wire frames) unit-tested on the VPS; the
    stage/coordinator GPU integration validated on the mesh.
  - **EAGLE-3 draft (2.2)** — ~2× tokens/round (0.389 → ~0.85 acceptance), zero added hops.
    Mostly download+integrate (pretrained Qwen2.5-72B head). `CIRCUIT_DRAFT_KIND=eagle`.
  - Amortize the spin-up: bring the 72B mesh up *once*, measure P1 proximity + chain + EAGLE
    together against the 1.5 tok/s cross-DC baseline.
- **Wave B — stack more tokens-per-round + kill prefill:**
  - **Latency-scaled spec trees (2.1)** — EAGLE-3 is tree-native; size the tree by RTT.
  - **Prefix KV cache (4.1)** — TTFT; deletes the prompt's network traversal for shared prefixes.
  - Async speculative pipelining (3.2) if EAGLE drafting sits on the critical path.
- **Wave C — the heavy / research bets:**
  - **Early-exit / adaptive depth (1.1)** — biggest avg-case win; quality-sensitive, behind a gate.
  - **Speculation swarm (2.3)** — per-node idle-compute drafts harvested in-flight; research bet.
  - **Fewest-fattest stages (1.2)** — lands with dynamic re-slicing.
- **Emergent (no build):** regional replication (1.3) appears with node density.

## Testing

`tc netem` injects per-link delay/jitter on a cheap throwaway mesh to emulate a geographic
spread (more controllable than real cross-DC placement). `topology.py`/`specdecode.py` stay
pure logic → deterministic unit tests over synthetic latency matrices and token trees, with
netem runs for end-to-end confirmation. Gate every technique behind an env flag (like
`CIRCUIT_BATCH`/`CIRCUIT_MAX_CONCURRENCY`) so the default path stays byte-identical until proven.
