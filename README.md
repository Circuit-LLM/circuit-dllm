# circuit-engine

The Python inference engine for the Circuit decentralized LLM — pipeline-parallel
stage workers + coordinator, joined over an encrypted wire, driven by
async-pipelined speculative decoding.

This is the compute path described in
`circuit-decentralized-llm/docs/BUILD_PLAN.md` and `TWO_L4_WAN_SPLIT.md`.

## Originality

All code here is written from scratch from our own design docs. It does **not**
copy or derive from any third-party distributed-inference codebase. The
techniques it uses — pipeline parallelism, speculative decoding, async
pipelining — are general, long-published methods, implemented here in our own
way. Standard libraries (PyTorch, Hugging Face `transformers`, `cryptography`)
are used as ordinary dependencies, the same as anyone would.

## Layout

```
engine/
  wire.py        encrypted length-prefixed framing (ChaCha20-Poly1305)   [no torch]
  tensors.py     tensor <-> bytes serialization                          [torch]
  kv.py          per-session KV cache with speculative rollback          [torch]
  stage.py       a pipeline stage worker (holds a contiguous layer block)[torch]
  coordinator.py embedding + lm_head + sampling + draft + orchestration   [torch]
tests/
  test_wire.py   wire round-trip + tamper-detection (CPU, no torch)
```

## Phases

See `BUILD_PLAN.md`. Phase 0 = foundations + byte-exact correctness vs a
single-process reference. Speed (CUDA graphs, async pipeline) is Phase 1.
