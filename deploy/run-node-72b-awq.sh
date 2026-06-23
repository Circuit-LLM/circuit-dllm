#!/usr/bin/env bash
# AWQ-per-node stage worker for the 72B mesh (docs/AWQ_PER_NODE.md, the FAST mesh path):
# stage the full 72B-AWQ once, slice THIS node's layer range into a complete sub-model, and
# serve it with AWQ/Marlin (2.1x bnb). Static mode (--layers + --key) — the speed measurement
# doesn't need the dynamic control plane.
#
# Env:
#   CIRCUIT_LAYERS     this node's slice, e.g. 16:48   (required)
#   CIRCUIT_KEY        64-hex wire key, must match the coordinator   (required)
#   CIRCUIT_MODEL      AWQ repo (default Qwen/Qwen2.5-72B-Instruct-AWQ)
#   CIRCUIT_PORT       stage listen port (default 19210)
set -eu
ENGINE_DIR="${CIRCUIT_ENGINE_DIR:-/root/circuit-engine}"
cd "$ENGINE_DIR"
export CIRCUIT_MODEL="${CIRCUIT_MODEL:-Qwen/Qwen2.5-72B-Instruct-AWQ}"
export MODEL_HOME="${MODEL_HOME:-/root/hf-cache}"
export HF_HOME="$MODEL_HOME" HF_HUB_DISABLE_TELEMETRY=1
: "${CIRCUIT_LAYERS:?set CIRCUIT_LAYERS=START:END}"
: "${CIRCUIT_KEY:?set CIRCUIT_KEY (64 hex)}"
START="${CIRCUIT_LAYERS%%:*}"; END="${CIRCUIT_LAYERS##*:}"
SUB_DIR="/root/sub-${START}-${END}"

if [ ! -f "$SUB_DIR/model.safetensors" ]; then
  echo "[awq-node] staging full AWQ + slicing layers [$START,$END) -> $SUB_DIR"
  FULL=$(python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('$CIRCUIT_MODEL'))")
  python3 "$ENGINE_DIR/scripts/slice-awq.py" "$FULL" "$SUB_DIR" "$START" "$END"
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

echo "[awq-node] serving sub-model [$START,$END) on :${CIRCUIT_PORT:-19210}"
exec python3 -u -m engine.stage_worker \
  --port "${CIRCUIT_PORT:-19210}" --model "$CIRCUIT_MODEL" --device cuda --host 0.0.0.0 \
  --layers "$START:$END" --key "$CIRCUIT_KEY" --submodel "$SUB_DIR"
