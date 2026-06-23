#!/usr/bin/env bash
# Coordinator for the 72B bnb-4bit mesh (role=coordinator-mesh, quantize-at-load).
#
# Unlike run-coordinator-mesh.sh (32B AWQ, model staged to /dev/shm), the 72B fp16 is
# ~144 GB — it can NOT fit in RAM, so we load it straight off the CONTAINER DISK
# (MODEL_HOME on the local SSD, mmap-safe; a network volume would FUSE-deadlock). The
# bnb loader (CIRCUIT_QUANT=bnb + CIRCUIT_SHARD=1) reads the fp16 and quantizes ONLY this
# pod's slice (head + CIRCUIT_LOCAL_LAYERS) to 4-bit in VRAM — the un-owned layers are
# never materialized. Joined nodes (run-node-72b.sh) cover the rest of [0,80).
#
# Why the coordinator MUST hold a slice: with no local layers it would call load_model()
# = the full 144 GB → won't fit one L4. So it always shard-loads head + a small slice.
set -u
ENGINE_DIR="${CIRCUIT_ENGINE_DIR:-/opt/circuit-engine}"
cd "$ENGINE_DIR" || exit 1

export CIRCUIT_MODEL="${CIRCUIT_MODEL:-Qwen/Qwen2.5-72B-Instruct}"   # fp16 SOURCE (bnb quantizes at load)
export CIRCUIT_DRAFT="${CIRCUIT_DRAFT:-Qwen/Qwen2.5-0.5B-Instruct}"  # coordinator holds the draft (fp16, tiny)

# Load in place off the container disk: pass DEST=MODEL_HOME so stage-model.sh downloads
# there and skips the /dev/shm copy (144 GB won't fit RAM). /root is always container disk.
export MODEL_HOME="${MODEL_HOME:-/root/hf-cache}"
bash "$ENGINE_DIR/deploy/stage-model.sh" "$MODEL_HOME" || exit 1
export HF_HOME="$MODEL_HOME"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1

export CIRCUIT_KEY="${CIRCUIT_KEY:-${CLUSTER_KEY:-$(cat /workspace/cluster_key.txt 2>/dev/null)}}"
export CIRCUIT_DEVICE=cuda
export CIRCUIT_API_PORT="${CIRCUIT_API_PORT:-18931}"
export CIRCUIT_MAX_CONCURRENCY="${CIRCUIT_MAX_CONCURRENCY:-4}"

# ── bnb 4-bit shard-load: head + this pod's co-located slice ──────────────────
export CIRCUIT_QUANT=bnb
export CIRCUIT_SHARD=1
export CIRCUIT_LOCAL_LAYERS="${CIRCUIT_LOCAL_LAYERS:-0:16}"   # coordinator holds layers 0:16 + head + draft

# ── mesh control plane ───────────────────────────────────────────────────────
export CIRCUIT_MESH=1
export CIRCUIT_MESH_LAYERS="${CIRCUIT_MESH_LAYERS:-80}"       # Qwen2.5-72B = 80 layers
export CIRCUIT_MESH_STAGES="${CIRCUIT_MESH_STAGES:-3}"        # 3 nodes split [16,80) -> ~21 layers each
export CIRCUIT_MESH_FP="${CIRCUIT_MESH_FP:-qwen2.5-72b-bnb}"  # must match nodes' --model-fp
export CIRCUIT_MESH_REPLICATION="${CIRCUIT_MESH_REPLICATION:-1}"
export CIRCUIT_CONTROL_HOST=0.0.0.0
export CIRCUIT_CONTROL_PORT="${CIRCUIT_CONTROL_PORT:-18932}"
export CIRCUIT_COORDINATOR_ADVERTISE="${CIRCUIT_COORDINATOR_ADVERTISE:-}"
export CIRCUIT_MESH_VERIFY_SIG="${CIRCUIT_MESH_VERIFY_SIG:-1}"

exec python3 -u -m engine.api
