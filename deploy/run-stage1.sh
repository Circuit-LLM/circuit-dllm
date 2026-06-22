#!/usr/bin/env bash
# pm2-runtime-managed stage 1 worker (role=stage1) — layers 32:64 of the 32B.
#
# Self-contained GPU node: deps + engine code baked into the image (code at
# /opt/circuit-engine, override with CIRCUIT_ENGINE_DIR). The model is provisioned
# onto THIS node's own volume (downloaded from HF if absent) and staged to local RAM
# on boot, then loaded from there (avoids the mfs FUSE-mmap hang). The only things
# shared with the coordinator are the cluster key (CIRCUIT_KEY) and the network link
# — this is the "someone hooks their GPU to the network" node.
ENGINE_DIR="${CIRCUIT_ENGINE_DIR:-/opt/circuit-engine}"
cd "$ENGINE_DIR" || exit 1

bash "$ENGINE_DIR/deploy/stage-model.sh" /dev/shm/hf-cache || exit 1
export HF_HOME=/dev/shm/hf-cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1

exec python3 -u -m engine.stage_worker \
  --port 19210 --layers 32:64 \
  --model Qwen/Qwen2.5-32B-Instruct-AWQ \
  --key "${CIRCUIT_KEY:-${CLUSTER_KEY:-$(cat /workspace/cluster_key.txt 2>/dev/null)}}" \
  --device cuda --host 0.0.0.0 --prune
