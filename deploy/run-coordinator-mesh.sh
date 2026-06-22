#!/usr/bin/env bash
# Coordinator in MESH mode (role=coordinator-mesh) — runs the embed/lm_head/draft + the
# OpenAI API + the control plane (Topology + Registry + /register control server), and
# accepts GPU nodes that join dynamically and get assigned a layer slice. This is the
# coordinator WE run; contributors run run-mesh.sh against its control URL.
#
# Differs from run-api.sh (static): no hand-wired CIRCUIT_STAGES; instead CIRCUIT_MESH=1
# turns on dynamic topology. By default the coordinator holds NO transformer layers
# (CIRCUIT_LOCAL_LAYERS unset → coordinator_end=0), so the full [0,LAYERS) is served by
# joined nodes. Set CIRCUIT_LOCAL_LAYERS=0:N to also hold a co-located slice on its GPU.
ENGINE_DIR="${CIRCUIT_ENGINE_DIR:-/opt/circuit-engine}"
cd "$ENGINE_DIR" || exit 1

export CIRCUIT_MODEL=Qwen/Qwen2.5-32B-Instruct-AWQ
export CIRCUIT_DRAFT=Qwen/Qwen2.5-0.5B-Instruct      # coordinator holds the draft (before stage-model)
bash "$ENGINE_DIR/deploy/stage-model.sh" /dev/shm/hf-cache || exit 1
export HF_HOME=/dev/shm/hf-cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1

export CIRCUIT_KEY="${CIRCUIT_KEY:-${CLUSTER_KEY:-$(cat /workspace/cluster_key.txt 2>/dev/null)}}"
export CIRCUIT_DEVICE=cuda
export CIRCUIT_API_PORT=18931
export CIRCUIT_MAX_CONCURRENCY="${CIRCUIT_MAX_CONCURRENCY:-4}"

# ── mesh control plane ───────────────────────────────────────────────────────
export CIRCUIT_MESH=1
export CIRCUIT_MESH_LAYERS="${CIRCUIT_MESH_LAYERS:-64}"
export CIRCUIT_MESH_STAGES="${CIRCUIT_MESH_STAGES:-3}"
export CIRCUIT_MESH_FP="${CIRCUIT_MESH_FP:-qwen2.5-32b-awq}"   # must match clients' --model-fp
export CIRCUIT_MESH_REPLICATION="${CIRCUIT_MESH_REPLICATION:-1}"
export CIRCUIT_CONTROL_HOST=0.0.0.0
export CIRCUIT_CONTROL_PORT="${CIRCUIT_CONTROL_PORT:-18932}"   # exposed so clients can /register
# CIRCUIT_COORDINATOR_ADVERTISE: public host:port nodes are told to reach the coordinator at.
export CIRCUIT_COORDINATOR_ADVERTISE="${CIRCUIT_COORDINATOR_ADVERTISE:-}"
# CIRCUIT_MESH_SECRET falls back to CIRCUIT_KEY (private net). Signature enforcement
# (CIRCUIT_MESH_VERIFY_SIG=1) is OFF until the client signs its registration — run
# permissioned on a private net until then (see docs/UNIFIED_NODE.md Phase 2).

exec python3 -u -m engine.api
