#!/usr/bin/env bash
# Mesh node for the 72B bnb-4bit mesh (role=mesh, quantize-at-load) — joins a coordinator
# and gets a layer slice over [coordinator_end, 80) assigned dynamically.
#
# Like run-mesh.sh but for the 72B fp16 (~144 GB): loads off the CONTAINER DISK, not RAM,
# and shard-loads 4-bit with `--shard --quant bnb` (vs run-mesh.sh's `--prune`, which
# loads the WHOLE model then prunes — impossible at 144 GB). Only this node's assigned
# layers land in VRAM (4-bit); the rest stay on meta.
#
# Env (set by the node-client supervisor, or per-pod for tests):
#   CIRCUIT_CONTROL_URL    coordinator control endpoint http://host:18932   (required)
#   CIRCUIT_NODE_KEY       ed25519 private key hex (optional; else generated/persisted)
#   CIRCUIT_MODEL_FP       must equal the coordinator's CIRCUIT_MESH_FP      (default qwen2.5-72b-bnb)
#   CIRCUIT_CAPACITY_LAYERS max contiguous layers this L4 can hold           (default 30)
#   CIRCUIT_ADVERTISE_HOST / CIRCUIT_ADVERTISE_PORT  public addr coordinator dials (auto on RunPod)
set -u
ENGINE_DIR="${CIRCUIT_ENGINE_DIR:-/opt/circuit-engine}"
cd "$ENGINE_DIR" || exit 1

export CIRCUIT_MODEL="${CIRCUIT_MODEL:-Qwen/Qwen2.5-72B-Instruct}"   # fp16 source; no draft on a stage node
export MODEL_HOME="${MODEL_HOME:-/root/hf-cache}"                    # container disk (mmap-safe; not a volume)
bash "$ENGINE_DIR/deploy/stage-model.sh" "$MODEL_HOME" || exit 1
export HF_HOME="$MODEL_HOME"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1

: "${CIRCUIT_CONTROL_URL:?CIRCUIT_CONTROL_URL (http://coord-host:18932) must be set}"

# Self-discover the PUBLIC address the coordinator dials (RunPod proxy mapping for :19210).
ADV_HOST="${CIRCUIT_ADVERTISE_HOST:-${RUNPOD_PUBLIC_IP:-}}"
ADV_PORT="${CIRCUIT_ADVERTISE_PORT:-${RUNPOD_TCP_PORT_19210:-0}}"
echo "[run-node-72b] advertising ${ADV_HOST:-<bind>}:${ADV_PORT:-<bind>} -> coordinator $CIRCUIT_CONTROL_URL"

exec python3 -u -m engine.stage_worker \
  --port 19210 --model "$CIRCUIT_MODEL" --device cuda --host 0.0.0.0 \
  --shard --quant bnb \
  --control-url   "$CIRCUIT_CONTROL_URL" \
  --node-key      "${CIRCUIT_NODE_KEY:-}" \
  --node-key-file "${CIRCUIT_NODE_KEY_FILE:-/root/node_key.hex}" \
  --region        "${CIRCUIT_REGION:-}" \
  --capacity-layers "${CIRCUIT_CAPACITY_LAYERS:-30}" \
  --model-fp      "${CIRCUIT_MODEL_FP:-qwen2.5-72b-bnb}" \
  --advertise-host "$ADV_HOST" \
  --advertise-port "$ADV_PORT" \
  --payout-wallet "${CIRCUIT_PAYOUT_WALLET:-}"
