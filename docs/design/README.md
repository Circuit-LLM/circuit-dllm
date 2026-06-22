# Design rationale & evolution

These are the **"why" docs** — the reasoning that led to the architecture Circuit
DLLM runs today. They were written during the prototype phase (in the now-retired
`circuit-decentralized-llm-retired` repo) and are preserved here because the design
decisions they argue still govern the live system. The other docs in `../` describe
*how it works now*; these explain *why it's built this way*.

Read them as a progression:

1. **[MESH_ARCHITECTURE.md](MESH_ARCHITECTURE.md)** — the one law that governs
   everything: *per-token latency ≈ (machine boundaries in the per-token loop) ×
   RTT + compute*. Why naive layer-splitting over the open internet is slow, and the
   capacity-pooling vs. throughput-replication tradeoff.
2. **[SPECULATION_FOREST.md](SPECULATION_FOREST.md)** — why speculative / predictive
   drafting is the lever: decode is memory-bandwidth-bound, so verifying K candidate
   tokens costs ~the same as verifying one.
3. **[TWO_L4_WAN_SPLIT.md](TWO_L4_WAN_SPLIT.md)** — the concrete design that became
   the current system: a 32B (4-bit) split 32/32 across two L4s over one WAN hop,
   served via async-pipelined speculative decoding. **This is essentially the
   architecture running in production today.**
4. **[BUILD_PLAN.md](BUILD_PLAN.md)** — the end-to-end roadmap: product, resilience,
   operations, decentralization, payments.

## What's moved on since these were written

The docs are kept faithful to their original form; a few specifics have since changed:

- **Draft model:** these explore a 3B draft (and a CPU draft "forest"); the live
  engine uses a single **0.5B** draft (`Qwen2.5-0.5B-Instruct`) co-located on the
  coordinator.
- **Payments:** referred to here as **PISKY**; the live settlement token is **CIRC**
  (via the x402 inference gateway).
- **The "1×L4 + 6 CPU clients / 7B" framing** in MESH/SPECULATION is the *problem*
  they argue away from — not the current build. The current build is the 2×L4 / 32B
  design of `TWO_L4_WAN_SPLIT.md`.

For the current implementation, ops, scaling and node-join, see:
[../ARCHITECTURE.md](../ARCHITECTURE.md) · [../OPERATING.md](../OPERATING.md) ·
[../SCALING.md](../SCALING.md) · [../JOIN.md](../JOIN.md) · [../BATCHING.md](../BATCHING.md)
