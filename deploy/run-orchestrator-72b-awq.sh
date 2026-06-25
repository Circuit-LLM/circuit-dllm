#!/usr/bin/env bash
# Head-only FLOATING ORCHESTRATOR for the 72B AWQ mesh (docs/FLOATING_COORDINATOR.md, CUTOVER_FLOATING.md).
# Slices a head-only AWQ submodel locally (embed/norm/lm_head, 0 layers ~4.7GB) + the 1.5B draft,
# registers with the standalone control plane, and serves /v1/chat/completions — relaying ALL 80
# layers to the holders (no co-located slice). Mirrors run-coordinator-72b-awq.sh minus the layer slice.
#
# Env: CIRCUIT_CONTROL_URL (required), CIRCUIT_KEY (required), CIRCUIT_ORCH_ADVERTISE (public ip the
#      gateway reaches), CIRCUIT_API_PORT (default 18931), CIRCUIT_MESH_FP (default qwen2.5-72b-awq).
set -eu
ENGINE_DIR="${CIRCUIT_ENGINE_DIR:-/workspace/ce}"
cd "$ENGINE_DIR"
export CIRCUIT_MODEL="${CIRCUIT_MODEL:-Qwen/Qwen2.5-72B-Instruct-AWQ}"
export MODEL_HOME="${MODEL_HOME:-/root/hf-cache}"; export HF_HOME="$MODEL_HOME" HF_HUB_DISABLE_TELEMETRY=1
: "${CIRCUIT_CONTROL_URL:?set CIRCUIT_CONTROL_URL=http://<control>:18932}"
: "${CIRCUIT_KEY:?set CIRCUIT_KEY (64 hex)}"

HEAD=/root/head-only
if [ ! -f "$HEAD/model.safetensors" ]; then
  echo "[awq-orch] staging full AWQ + slicing head-only (0 layers) -> $HEAD"
  FULL=$(python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('$CIRCUIT_MODEL', allow_patterns=['*.safetensors','*.json','*.txt']))")
  python3 "$ENGINE_DIR/scripts/slice-awq.py" "$FULL" "$HEAD" 0 0 --keep-head
fi
DRAFT_ID="${CIRCUIT_DRAFT_ID:-Qwen/Qwen2.5-1.5B-Instruct}"
export CIRCUIT_DRAFT="$(python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('$DRAFT_ID'))")"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

export CIRCUIT_ROLE=orchestrator
export CIRCUIT_COORD_SUBMODEL="$HEAD"        # head-only AWQ submodel (local_layers stays None)
export CIRCUIT_DEVICE=cuda
export CIRCUIT_API_PORT="${CIRCUIT_API_PORT:-18931}"
export CIRCUIT_MESH_FP="${CIRCUIT_MESH_FP:-qwen2.5-72b-awq}"
export CIRCUIT_MAX_CONCURRENCY="${CIRCUIT_MAX_CONCURRENCY:-4}"
echo "[awq-orch] head-only orchestrator -> control $CIRCUIT_CONTROL_URL, advertise ${CIRCUIT_ORCH_ADVERTISE:-<bind>}:$CIRCUIT_API_PORT"
exec python3 -u -m engine.api
