#!/usr/bin/env bash
# circuit-gpu-node — container entrypoint. Joins the live mesh using the engine's proven
# control-client path (same code the production pods run). Operator sets a couple of env vars;
# this handles GPU detection, capacity sizing, identity persistence, and address advertisement.
set -u

say() { echo "[circuit-gpu-node] $*"; }
die() { echo "[circuit-gpu-node] ERROR: $*" >&2; exit 1; }

ENGINE_DIR="${CIRCUIT_ENGINE_DIR:-/opt/circuit-engine}"
STATE_DIR="${CIRCUIT_STATE_DIR:-/var/lib/circuit}"
cd "$ENGINE_DIR" || die "engine dir $ENGINE_DIR missing (bad image?)"

# ── required / defaulted config ───────────────────────────────────────────────────────────
: "${CIRCUIT_CONTROL_URL:?set CIRCUIT_CONTROL_URL — the coordinator control endpoint, e.g. https://join.circuitllm.xyz}"
export CIRCUIT_MODEL="${CIRCUIT_MODEL:-Qwen/Qwen2.5-72B-Instruct-AWQ}"
export CIRCUIT_MODEL_FP="${CIRCUIT_MODEL_FP:-qwen2.5-72b-awq}"
export CIRCUIT_AWQ_SHARDS="${CIRCUIT_AWQ_SHARDS:-circuitllmdev/qwen2.5-72b-awq-shards}"
export HF_HOME="${HF_HOME:-$STATE_DIR/hf-cache}"
export HF_HUB_DISABLE_TELEMETRY=1
PORT="${CIRCUIT_PORT:-19210}"

# Persist identity + model cache on the mounted volume. Without a volume the node would
# re-register as a brand-new id and re-download its shards on every restart.
mkdir -p "$STATE_DIR" "$HF_HOME" || die "cannot write state dir $STATE_DIR (mount a volume at $STATE_DIR)"
KEY_FILE="${CIRCUIT_NODE_KEY_FILE:-$STATE_DIR/node_key.hex}"
export CIRCUIT_NODE_KEY_FILE="$KEY_FILE"
if [ ! -f "$KEY_FILE" ]; then
  say "first run — a stable node identity will be generated and saved to $KEY_FILE"
fi

# ── GPU preflight ─────────────────────────────────────────────────────────────────────────
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
  die "no NVIDIA GPU visible inside the container.
       Run with '--gpus all' and ensure the NVIDIA Container Toolkit is installed on the host.
       (The installer sets this up for you; see https://circuitllm.xyz/join )"
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
VRAM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)

# Capacity = how many contiguous layers this card can hold. ~0.5 GB/layer for 72B-AWQ, minus a
# ~4 GB headroom for activations/KV. The coordinator assigns a slice within this budget.
if [ -z "${CIRCUIT_CAPACITY_LAYERS:-}" ]; then
  if [ -n "${VRAM_MIB:-}" ]; then
    CIRCUIT_CAPACITY_LAYERS=$(awk "BEGIN{c=int(($VRAM_MIB/1024 - 4)/0.5); if(c>80)c=80; if(c<1)c=1; print c}")
  else
    CIRCUIT_CAPACITY_LAYERS=40
  fi
fi
export CIRCUIT_CAPACITY_LAYERS
[ "${VRAM_MIB:-0}" -lt 8192 ] 2>/dev/null && say "WARNING: ${VRAM_MIB:-?} MiB VRAM is small for a 72B slice; the node will hold few layers."

# ── address the coordinator will dial ─────────────────────────────────────────────────────
# Cloud/public-IP nodes advertise their reachable host:port. RunPod exposes these via env.
# Home/NAT nodes use the relay (Phase 3): CIRCUIT_RELAY_URL → advertise the relay endpoint.
ADV_HOST="${CIRCUIT_ADVERTISE_HOST:-${RUNPOD_PUBLIC_IP:-}}"
ADV_PORT="${CIRCUIT_ADVERTISE_PORT:-${RUNPOD_TCP_PORT_19210:-0}}"
if [ -z "$ADV_HOST" ] && [ -z "${CIRCUIT_RELAY_URL:-}" ]; then
  say "NOTE: no CIRCUIT_ADVERTISE_HOST and no CIRCUIT_RELAY_URL set."
  say "      A public-IP host needs CIRCUIT_ADVERTISE_HOST; a home/NAT host needs the relay."
fi

say "GPU: ${GPU_NAME:-unknown} (${VRAM_MIB:-?} MiB) → capacity ${CIRCUIT_CAPACITY_LAYERS} layers"
say "joining $CIRCUIT_CONTROL_URL  |  model_fp=$CIRCUIT_MODEL_FP  |  shards=$CIRCUIT_AWQ_SHARDS"
say "advertise ${ADV_HOST:-<relay/bind>}:${ADV_PORT:-<bind>}  |  identity $KEY_FILE"

exec python3 -u -m engine.stage_worker \
  --port "$PORT" --model "$CIRCUIT_MODEL" --device cuda --host 0.0.0.0 \
  --control-url     "$CIRCUIT_CONTROL_URL" \
  --node-key        "${CIRCUIT_NODE_KEY:-}" \
  --node-key-file   "$KEY_FILE" \
  --region          "${CIRCUIT_REGION:-}" \
  --capacity-layers "$CIRCUIT_CAPACITY_LAYERS" \
  --model-fp        "$CIRCUIT_MODEL_FP" \
  --advertise-host  "$ADV_HOST" \
  --advertise-port  "$ADV_PORT" \
  --payout-wallet   "${CIRCUIT_PAYOUT_WALLET:-}"
