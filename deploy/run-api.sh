#!/usr/bin/env bash
# PM2/pm2-runtime-managed API + co-located stage 0 (role=api) — MESH-less static.
#
# Self-contained node: deps + engine code are baked into the pod image. The engine
# code lives at /opt/circuit-engine (override with CIRCUIT_ENGINE_DIR for dev pods
# that sync code to the volume). The model is provisioned onto THIS node's own
# volume (downloaded from HF if absent) and staged to local RAM on boot, then loaded
# from there (avoids the mfs FUSE-mmap hang). Nothing is shared with other nodes
# except the cluster key (CIRCUIT_KEY) and the peer address (CIRCUIT_STAGES) — both
# passed in via the pod env, like a real independent operator joining the network.
ENGINE_DIR="${CIRCUIT_ENGINE_DIR:-/opt/circuit-engine}"
cd "$ENGINE_DIR" || exit 1

# Model + draft MUST be exported before stage-model.sh runs — stage-model reads
# CIRCUIT_DRAFT to decide whether to also provision the 0.5B draft. (If they're set
# after, stage-model skips the draft, then the engine loads offline and crashes
# looking for it.)
export CIRCUIT_MODEL=Qwen/Qwen2.5-32B-Instruct-AWQ
export CIRCUIT_DRAFT=Qwen/Qwen2.5-0.5B-Instruct

bash "$ENGINE_DIR/deploy/stage-model.sh" /dev/shm/hf-cache || exit 1
export HF_HOME=/dev/shm/hf-cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1

export CIRCUIT_KEY="${CIRCUIT_KEY:-${CLUSTER_KEY:-$(cat /workspace/cluster_key.txt 2>/dev/null)}}"
export CIRCUIT_STAGES="${CIRCUIT_STAGES:?CIRCUIT_STAGES (pod2 host:port) must be set in the pod env}"
export CIRCUIT_LOCAL_LAYERS=0:32
export CIRCUIT_DEVICE=cuda
export CIRCUIT_API_PORT=18931
export CIRCUIT_MAX_CONCURRENCY=4

exec python3 -u -m engine.api
