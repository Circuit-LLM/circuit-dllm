# Circuit DLLM — Architecture v3: "Speculation Forest"

> How to actually get to usable token rates with the hardware we have
> (1× L4 GPU + 6× CPU VPS clients) by combining **replication** (whole model
> on the GPU) with a **distributed speculation forest** (the CPUs become a
> parallel drafting swarm) and **latency hiding** (the network round trip is
> overlapped, never in the critical path).
>
> Builds on [MESH_ARCHITECTURE.md](MESH_ARCHITECTURE.md). Nothing here is built
> yet — this is the design map + honest speed estimates.

---

## 0. The two laws

1. **Per-token latency = (machine boundaries in the per-token loop) × RTT + compute.**
   → Keep the heavy model's forward pass on ONE machine. Network out of the loop.
2. **Decode is memory-bandwidth-bound, not compute-bound.** Loading the 4.7 GB of
   weights once dominates a forward pass. **Verifying K candidate tokens in a tree
   costs almost the same as verifying 1** (same weight read, reused across the K
   positions). → This is *why* speculation wins: many tokens, ~one weight load.

The whole design follows from these: **put the big model whole on the GPU
(law 1), and harvest many tokens per weight-load by verifying a wide tree the
CPUs drafted (law 2).**

---

## 1. The inversion (vs. v1)

| | v1 (split pipeline) | v3 (speculation forest) |
|---|---|---|
| What the CPU holds | a **slice of the big model's layers** | a **whole small draft model** (0.5B) |
| What crosses the network | hidden-state activations, **per token** | **tokens** (ints), best-effort, **out of the loop** |
| CPU topology | a **chain** (links) — 6 sequential hops | a **fan** (leaves) — 6 parallel, independent |
| Big-model forward pass | crosses 7 machines | stays **local** to the GPU |
| Network in per-token critical path? | **yes** → ~3 s/token | **no** → GPU-local speed |

The CPU holds a small *model*, not a small *shard*. That one change moves the
network out of the per-token loop.

---

## 2. Components — what holds what

```
┌──────────────────────────── GPU NODE (L4) ────────────────────────────┐
│  • FULL Qwen2.5-7B  (target / verifier)            ~4.7 GB VRAM         │
│  • TINY 0.5B draft  (LOCAL — guarantees the floor)  ~0.4 GB VRAM        │
│  • Tree verifier (tree-attention, accept/rollback)                     │
│  • Continuous speculation scheduler                                     │
│  → runs full speculative decoding ENTIRELY LOCALLY (no network needed)  │
└───────────────────────────────────────────────────────────────────────┘
        ▲ best-effort token branches (tiny msgs)   │ accepted-path feedback
        │ (used IF they arrive in time, else ignored)▼
┌─────────────── CPU SWARM (6× VPS, ~140 ms RTT, parallel) ──────────────┐
│  node1: 0.5B draft  ┐                                                  │
│  node2: 0.5B draft  │  each continuously drafts DIVERSE branches       │
│  node3: 0.5B draft  │  (different temps / offsets / continuations)     │
│  node4: 0.5B draft  │  → widen the verification tree opportunistically │
│  node5: retrieval   │  (REST: draft by datastore lookup, no model)     │
│  node6: retrieval   ┘                                                  │
└───────────────────────────────────────────────────────────────────────┘
┌──────────────────────────── ROUTER / REGISTRY ────────────────────────┐
│  tracks GPU verifiers (model, load, region) + drafters (quality)       │
│  routes each user request to a GPU; matches drafters to verifiers      │
│  x402 payment: verifier (compute) + accepted-branch drafters + router  │
└───────────────────────────────────────────────────────────────────────┘
```

**Key robustness property:** the GPU node runs spec decoding *fully locally*
using its own 0.5B draft. The remote CPU branches are **bonus** — used when they
arrive in time, ignored when late/disconnected. **Remote contributions can only
help, never hurt.** The network is never on the critical path.

---

## 3. How a request runs

```
1. user → router (verify x402) → pick a GPU node holding the model
2. GPU node prefills the prompt locally (one pass)              → TTFT ~0.5–1.5 s
3. DECODE LOOP (all local to the GPU; CPUs feed it async):
   a. local 0.5B draft proposes a branch;
      remote CPU drafts (already in flight) add MORE branches  → a token TREE
   b. GPU verifies the whole tree in ONE forward pass (law 2)
   c. accept the deepest matching path → commit A tokens
   d. roll KV forward by A; meanwhile CPUs are ALREADY drafting
      the next batch off the predicted path                    → RTT HIDDEN
4. tokens stream out through the router to the user as they commit
```

The expensive resources (GPU verify) stay 100% busy on useful work. A wrong
remote guess just wastes a cheap CPU draft — the misprediction penalty is borne
by the cheap, parallel side.

---

## 4. Estimated speeds (single stream)

Throughput ≈ **A / T_verify**, where A = tokens accepted per pass,
T_verify ≈ one weight-load (~25–35 ms on the L4 for 7B Q4).

| Configuration | Accepted/pass (A) | Est. tok/s | Notes |
|---|---|---|---|
| **Today** — split pipeline, 7 hops | 1 | **~0.3** | 3 s/token, network-bound |
| GPU solo, whole model, no spec | 1 | **~30–45** | memory-bandwidth ceiling |
| GPU + **local** 0.5B draft (spec) | ~2.5–3.5 | **~60–90** | guaranteed floor, no network |
| GPU + local draft + **CPU forest** (latency hidden) | ~3.5–5 | **~80–120** | CPUs widen tree; best-effort |
| GPU + a **3B draft on a joined GPU** | ~4–6 | **~110–150** | better draft → higher acceptance |

TTFT (time to first token): **~0.5–1.5 s** (one local prefill pass) in every
config above — vs. **minutes** today.

**Honest caveats**
- These assume good draft↔target agreement. Acceptance is content-dependent:
  high on code/repetitive/templated text, lower on hard/creative text.
- The CPU-forest row's upside depends on remote branches arriving within the
  ~25–35 ms verify window often enough to matter. They won't always — that's why
  the local draft sets a hard floor (~60–90) the remote swarm can only raise.
- **Naïve (synchronous) distributed spec decoding would be RTT-bound and SLOWER
  than GPU-solo** (140 ms RTT ≫ 30 ms verify). Latency hiding + a local-draft
  floor is what makes the distributed part safe. This is the load-bearing detail.

---

## 5. What happens when a new GPU client joins

A joining GPU announces its capability to the registry and slots into one of
three roles (its choice / the registry's based on need):

### Role A — **Verifier replica** (holds the whole model)
- Loads the full 7B, registers as another verifier.
- Router load-balances user requests across all verifiers.
- **Effect: +1× concurrent throughput.** N verifiers ≈ N× simultaneous users,
  each still at full per-stream speed. **Per-stream latency unchanged; capacity
  scales linearly.** No re-sharding, no coordination — pure BitTorrent-seeder add.
- Each verifier runs its own local spec decoding and can recruit drafters.

### Role B — **High-quality drafter** (better draft than the CPU 0.5Bs)
- A mid GPU runs a 3B–7B draft fast → much higher acceptance (A) than 0.5B CPU
  drafts → **raises the tok/s of the verifier(s) it feeds** (see table row 5).
- Effect: doesn't add concurrency, but makes existing streams *faster*.

### Role C — **Co-located split partner** (for models too big to fit one GPU)
- Pairs with another GPU over fast local interconnect to host, e.g., a 70B as
  one logical verifier. Only used for the too-big-to-replicate case.

### The scaling picture
```
        more VERIFIER GPUs  → more users served at once   (throughput ↑, linear)
        better DRAFTER GPUs → each stream runs faster      (latency ↓)
        more CPU drafters   → wider trees, opportunistic    (acceptance ↑, free)
        co-located GPU pair → bigger models become possible (capability ↑)
```
The network is a **two-sided market**: verifiers (compute) and drafters
(acceleration), matched by the registry, paid per accepted work. Every kind of
hardware has a productive slot, and **adding any of them makes the network
strictly better** — the opposite of v1, where adding a CPU shard added a hop and
slowed every request.

---

## 6. Why this is "new territory"

The building blocks are real and separately proven on single machines:
- Tree / forest speculation + tree attention — SpecInfer, Medusa, Sequoia.
- Retrieval / n-gram drafting — REST, Lookahead decoding.
- Asynchronous / pipelined speculation — PEARL and related.

What's **unproven** is the synthesis we'd be attempting: doing the tree drafting
across **network-separated, decentralized machines** and **hiding WAN RTT** so a
swarm of cheap CPUs supercharges a single GPU past its solo rate — with a
local-draft floor guaranteeing no regression. Published tree-spec is single-box.
A WAN speculation forest is the genuine research bet.

---

## 7. First experiment (cheapest honest test, existing hardware)

1. **Full 7B on the L4 + local 0.5B draft, tree-verify, all local.** Measure.
   Target: clear **~60 tok/s** (proves replication + local spec).
2. **Attach 1–2 CPU drafters** feeding branches best-effort. Measure whether
   acceptance (A) rises and tok/s climbs toward **~80–100** — and confirm that
   when the CPUs are killed, speed *falls back to the floor*, never below.
3. If both hold, build the registry/router and test a **2nd GPU verifier** for
   concurrent-user throughput.

Step 1 alone tells you if the foundation is real; step 2 tells you if the
decentralized swarm actually adds value; step 3 proves the scaling story.

---

## 8. Bottom line

- **Replication** puts the network out of the per-token loop → ~30–45 tok/s
  baseline on the one L4 we already have.
- **Local speculative decoding** harvests multiple tokens per weight-load →
  ~60–90 tok/s, guaranteed, no network dependency.
- **The CPU speculation forest** widens the tree opportunistically → ~80–120
  tok/s, while giving all 6 CPUs a real, paid, decentralized role — and it can
  only help, never hurt.
- **New GPUs** join as verifiers (throughput ↑, linear), drafters (latency ↓),
  or co-located partners (bigger models) — every addition makes the network
  strictly better.

This is how the pieces we have, plus speculative decoding, plausibly get us from
**0.3 tok/s to 60–120 tok/s** — from "demo" to "usable product" — without
abandoning decentralization.
