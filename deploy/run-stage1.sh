#!/usr/bin/env bash
# pm2-runtime-managed stage 1 worker (pod 2, role=stage1) — layers 32:64 of the 32B.
# Deps baked into the image; model staged to local RAM on boot then loaded from there
# (avoids the mfs FUSE-mmap hang); engine code on the /workspace volume.
cd /workspace/circuit-engine || exit 1

bash /workspace/circuit-engine/deploy/stage-model.sh /dev/shm/hf-cache || exit 1
export HF_HOME=/dev/shm/hf-cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1

exec python3 -u -m engine.stage_worker \
  --port 19210 --layers 32:64 \
  --model Qwen/Qwen2.5-32B-Instruct-AWQ \
  --key "${CIRCUIT_KEY:-${CLUSTER_KEY:-$(cat /workspace/cluster_key.txt 2>/dev/null)}}" \
  --device cuda --host 0.0.0.0 --prune
