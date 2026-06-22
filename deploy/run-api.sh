#!/usr/bin/env bash
# PM2/pm2-runtime-managed API + co-located stage 0 (pod 1, role=api) — MESH-less static.
# Deps are baked into the pod image (system python3). Model is staged to local RAM on
# boot, then loaded from there (avoids the mfs FUSE-mmap hang). Engine code lives on
# the /workspace volume. CIRCUIT_STAGES (pod 2 address) comes from the pod env.
cd /workspace/circuit-engine || exit 1

bash /workspace/circuit-engine/deploy/stage-model.sh /dev/shm/hf-cache || exit 1
export HF_HOME=/dev/shm/hf-cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1

export CIRCUIT_MODEL=Qwen/Qwen2.5-32B-Instruct-AWQ
export CIRCUIT_KEY="${CIRCUIT_KEY:-${CLUSTER_KEY:-$(cat /workspace/cluster_key.txt 2>/dev/null)}}"
export CIRCUIT_STAGES="${CIRCUIT_STAGES:?CIRCUIT_STAGES (pod2 host:port) must be set in the pod env}"
export CIRCUIT_LOCAL_LAYERS=0:32
export CIRCUIT_DEVICE=cuda
export CIRCUIT_API_PORT=18931
export CIRCUIT_DRAFT=Qwen/Qwen2.5-0.5B-Instruct
export CIRCUIT_MAX_CONCURRENCY=4

exec python3 -u -m engine.api
