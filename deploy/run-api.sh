#!/usr/bin/env bash
# PM2-managed API + co-located stage 0 (pod 1): embed/lm_head + layers 0:32,
# OpenAI endpoint on :18931, connects to stage 1 (pod 2) at host:43638.
cd /workspace/circuit-engine || exit 1
export HF_HOME=/workspace/hf-cache
export CIRCUIT_MODEL=Qwen/Qwen2.5-32B-Instruct-AWQ
export CIRCUIT_KEY="$(cat /workspace/cluster_key.txt)"
export CIRCUIT_STAGES=157.157.221.29:43638
export CIRCUIT_LOCAL_LAYERS=0:32
export CIRCUIT_DEVICE=cuda
export CIRCUIT_API_PORT=18931
# Speculative-decode draft: the 0.5B (same Qwen2.5 family/tokenizer) proposes K
# tokens that the split 32B verifies in one pipeline round-trip. Output is
# token-identical to greedy; this only speeds up decode. Unset CIRCUIT_DRAFT to
# fall back to plain greedy. (~1GB extra VRAM on this pod.)
export CIRCUIT_DRAFT=Qwen/Qwen2.5-0.5B-Instruct
exec python3 -u -m engine.api
