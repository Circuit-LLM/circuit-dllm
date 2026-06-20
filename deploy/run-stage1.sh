#!/usr/bin/env bash
# PM2-managed stage 1 worker (pod 2): layers 32:64 of the 32B, sharded onto GPU.
# Reads the shared cluster key from /workspace so restarts reconnect.
cd /workspace/circuit-engine || exit 1
export HF_HOME=/workspace/hf-cache
exec python3 -u -m engine.stage_worker \
  --port 19210 --layers 32:64 \
  --model Qwen/Qwen2.5-32B-Instruct-AWQ \
  --key "$(cat /workspace/cluster_key.txt)" \
  --device cuda --host 0.0.0.0 --prune
