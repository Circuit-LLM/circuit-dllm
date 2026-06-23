#!/usr/bin/env bash
# AWQ-per-node coordinator for the 72B mesh (docs/AWQ_PER_NODE.md, the FAST mesh path):
# stage the full 72B-AWQ, slice a KEEP-HEAD coordinator sub-model [0,K) (embed + norm + lm_head
# + layers 0..K-1) and serve the OpenAI API, relaying the remaining layers to the stage nodes
# in CIRCUIT_STAGES. AWQ/Marlin throughout (2.1x bnb). Static stages (speed measurement).
#
# Env:
#   CIRCUIT_COORD_LAYERS  coordinator's head slice, e.g. 0:16   (default 0:16)
#     For a HETEROGENEOUS fleet, size slices ∝ GPU bandwidth so the slow card isn't the straggler:
#       python3 -m engine.topology layout <num_layers> "<coord-gpu>" "<stage-gpu>" ...
#     prints the CIRCUIT_COORD_LAYERS + per-stage CIRCUIT_LAYERS to use (SPEED_ROADMAP §1.2b).
#   CIRCUIT_STAGES        comma list host:port of the stage nodes   (required)
#   CIRCUIT_KEY           64-hex wire key, must match the nodes      (required)
#   CIRCUIT_MODEL         AWQ repo (default Qwen/Qwen2.5-72B-Instruct-AWQ)
#   CIRCUIT_DRAFT         draft for predictive drafting (default Qwen/Qwen2.5-1.5B-Instruct fp16,
#                         ~3GB, shares the Qwen2.5 tokenizer); staged + resolved to a local dir
#   CIRCUIT_API_PORT      OpenAI API port (default 18931)
set -eu
ENGINE_DIR="${CIRCUIT_ENGINE_DIR:-/root/circuit-engine}"
cd "$ENGINE_DIR"
export CIRCUIT_MODEL="${CIRCUIT_MODEL:-Qwen/Qwen2.5-72B-Instruct-AWQ}"
export MODEL_HOME="${MODEL_HOME:-/root/hf-cache}"
export HF_HOME="$MODEL_HOME" HF_HUB_DISABLE_TELEMETRY=1
: "${CIRCUIT_STAGES:?set CIRCUIT_STAGES=host:port,...}"
: "${CIRCUIT_KEY:?set CIRCUIT_KEY (64 hex)}"
CL="${CIRCUIT_COORD_LAYERS:-0:16}"
K="${CL##*:}"
SUB_DIR="/root/sub-coord-0-${K}"

if [ ! -f "$SUB_DIR/model.safetensors" ]; then
  echo "[awq-coord] staging full AWQ + slicing keep-head [0,$K) -> $SUB_DIR"
  FULL=$(python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('$CIRCUIT_MODEL'))")
  python3 "$ENGINE_DIR/scripts/slice-awq.py" "$FULL" "$SUB_DIR" 0 "$K" --keep-head
fi

# Stage the draft (predictive drafting) and resolve to its ABSOLUTE snapshot dir BEFORE going
# offline. Loading the draft from the bare repo id under HF_HUB_OFFLINE=1 flakes (transformers'
# offline cache-resolution raises "couldn't connect ... not in cache" even when it IS cached and
# HF is reachable); loading from the resolved local dir is reliable. Skip if already a path.
DRAFT_ID="${CIRCUIT_DRAFT:-Qwen/Qwen2.5-1.5B-Instruct}"
if [ -d "$DRAFT_ID" ]; then
  export CIRCUIT_DRAFT="$DRAFT_ID"
else
  echo "[awq-coord] staging draft $DRAFT_ID"
  export CIRCUIT_DRAFT="$(python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('$DRAFT_ID'))")"
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

export CIRCUIT_COORD_SUBMODEL="$SUB_DIR"
export CIRCUIT_LOCAL_LAYERS="0:$K"
export CIRCUIT_DEVICE=cuda
export CIRCUIT_API_PORT="${CIRCUIT_API_PORT:-18931}"
export CIRCUIT_MAX_CONCURRENCY="${CIRCUIT_MAX_CONCURRENCY:-4}"             # pipeline overlap for concurrency
# NO CIRCUIT_SHARD/CIRCUIT_QUANT — the submodel path supersedes them.
echo "[awq-coord] head [0,$K) + API :$CIRCUIT_API_PORT -> stages $CIRCUIT_STAGES"
exec python3 -u -m engine.api
