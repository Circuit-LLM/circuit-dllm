#!/usr/bin/env bash
# run-contributor-node.sh — the one command a CONTRIBUTOR runs to join the Circuit mesh with a GPU
# and serve a fast AWQ slice (docs/AWQ_PER_NODE.md productization). It joins the coordinator, gets
# assigned a layer range, and pulls ONLY that ~16GB slice from the published shard repo (no 40GB
# full-model download) — then serves it with Marlin (2.1x bnb).
#
# Unlike run-mesh.sh (which stages the whole model and --prunes), this uses the AWQ-per-node path:
# no full pre-stage; engine.shard_fetch.resolve_submodel downloads the assigned slice (or, only if
# the repo lacks it, slices locally from the full checkpoint).
#
# Env:
#   CIRCUIT_CONTROL_URL    coordinator control endpoint http://host:18932            (required)
#   CIRCUIT_AWQ_SHARDS     published shard repo to pull the slice from, or 'local'   (required)
#                          e.g. Circuit-LLM/qwen2.5-72b-awq-shards
#   CIRCUIT_MODEL          AWQ repo (model_fp source + local-slice fallback)         (default 72B-AWQ)
#   CIRCUIT_MODEL_FP       must equal the coordinator's CIRCUIT_MESH_FP              (default qwen2.5-72b-awq)
#   CIRCUIT_CAPACITY_LAYERS max contiguous layers this GPU can hold (auto from VRAM if unset)
#   CIRCUIT_NODE_KEY[_FILE] ed25519 identity (signs /register; persisted if not given)
#   CIRCUIT_REGION         coarse geo label for proximity routing                   (optional)
#   CIRCUIT_PAYOUT_WALLET  Solana wallet earnings settle to                         (optional)
set -u
ENGINE_DIR="${CIRCUIT_ENGINE_DIR:-/opt/circuit-engine}"
cd "$ENGINE_DIR" || exit 1
export CIRCUIT_MODEL="${CIRCUIT_MODEL:-Qwen/Qwen2.5-72B-Instruct-AWQ}"
export HF_HOME="${HF_HOME:-/root/hf-cache}" HF_HUB_DISABLE_TELEMETRY=1
: "${CIRCUIT_CONTROL_URL:?set CIRCUIT_CONTROL_URL (http://coord-host:18932)}"
: "${CIRCUIT_AWQ_SHARDS:?set CIRCUIT_AWQ_SHARDS (published shard repo, or 'local')}"
export CIRCUIT_AWQ_SHARDS
export CIRCUIT_WORK_DIR="${CIRCUIT_WORK_DIR:-/root}"

# Capacity: layers this GPU can hold. 72B-AWQ 4-bit ≈ 0.5 GB/layer; reserve ~4 GB for CUDA/KV/Marlin.
# Auto-derive from free VRAM unless the operator sets it. Capped at the model's layer count.
if [ -z "${CIRCUIT_CAPACITY_LAYERS:-}" ]; then
  VRAM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
  if [ -n "${VRAM_MIB:-}" ]; then
    CIRCUIT_CAPACITY_LAYERS=$(awk "BEGIN{c=int(($VRAM_MIB/1024 - 4)/0.5); if(c>80)c=80; if(c<1)c=1; print c}")
  else
    CIRCUIT_CAPACITY_LAYERS=40
  fi
fi
export CIRCUIT_CAPACITY_LAYERS

# Self-discover the PUBLIC address the coordinator dials (RunPod proxy maps our :19210); explicit
# CIRCUIT_ADVERTISE_* wins (a home/bare-metal box with its own public IP).
ADV_HOST="${CIRCUIT_ADVERTISE_HOST:-${RUNPOD_PUBLIC_IP:-}}"
ADV_PORT="${CIRCUIT_ADVERTISE_PORT:-${RUNPOD_TCP_PORT_19210:-0}}"
echo "[contributor] join $CIRCUIT_CONTROL_URL | shards=$CIRCUIT_AWQ_SHARDS | capacity=${CIRCUIT_CAPACITY_LAYERS} layers | advertise ${ADV_HOST:-<bind>}:${ADV_PORT:-<bind>}"

# No --prune / --shard / --submodel: the control client resolves the AWQ slice for the assigned
# range via CIRCUIT_AWQ_SHARDS after /register.
exec python3 -u -m engine.stage_worker \
  --port 19210 --model "$CIRCUIT_MODEL" --device cuda --host 0.0.0.0 \
  --control-url   "$CIRCUIT_CONTROL_URL" \
  --node-key      "${CIRCUIT_NODE_KEY:-}" \
  --node-key-file "${CIRCUIT_NODE_KEY_FILE:-/root/node_key.hex}" \
  --region        "${CIRCUIT_REGION:-}" \
  --capacity-layers "$CIRCUIT_CAPACITY_LAYERS" \
  --model-fp      "${CIRCUIT_MODEL_FP:-qwen2.5-72b-awq}" \
  --advertise-host "$ADV_HOST" \
  --advertise-port "$ADV_PORT" \
  --payout-wallet "${CIRCUIT_PAYOUT_WALLET:-}"
