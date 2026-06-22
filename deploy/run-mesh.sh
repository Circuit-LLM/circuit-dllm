#!/usr/bin/env bash
# Mesh client bringup (role=mesh) — a GPU node that JOINS a coordinator's mesh and is
# assigned a layer slice dynamically (vs run-stage1.sh's hand-wired --layers).
#
# Self-contained like the other run scripts: baked engine code + model self-provisioned
# to this node's own volume + staged to RAM. In mesh mode the worker needs NO --key
# (the per-node session key is derived by the coordinator and returned at /register);
# it does need a node id, a model fingerprint that matches the coordinator's, and — on
# a NAT/proxy like RunPod — the PUBLIC address the coordinator should dial to reach it.
#
# Env (set by the node-client supervisor, or per-pod for tests):
#   CIRCUIT_CONTROL_URL    coordinator control endpoint, e.g. http://1.2.3.4:18932   (required)
#   CIRCUIT_NODE_KEY       ed25519 private key hex; node id derives from it & signs /register (optional)
#   CIRCUIT_NODE_KEY_FILE  where the node key is persisted/generated (default /workspace/node_key.hex)
#   CIRCUIT_MODEL_FP       must equal the coordinator's CIRCUIT_MESH_FP                (default qwen2.5-32b-awq)
#   CIRCUIT_CAPACITY_LAYERS max contiguous layers this GPU can hold                    (default 64)
#   CIRCUIT_ADVERTISE_HOST / CIRCUIT_ADVERTISE_PORT  public host:port the coordinator dials (proxy)
#   CIRCUIT_PAYOUT_WALLET  Solana wallet earnings settle to                            (optional)
ENGINE_DIR="${CIRCUIT_ENGINE_DIR:-/opt/circuit-engine}"
cd "$ENGINE_DIR" || exit 1

export CIRCUIT_MODEL="${CIRCUIT_MODEL:-Qwen/Qwen2.5-32B-Instruct-AWQ}"
# No CIRCUIT_DRAFT here → stage-model fetches only the 32B (a stage node needs no draft).
bash "$ENGINE_DIR/deploy/stage-model.sh" /dev/shm/hf-cache || exit 1
export HF_HOME=/dev/shm/hf-cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1

: "${CIRCUIT_CONTROL_URL:?CIRCUIT_CONTROL_URL (http://coord-host:18932) must be set}"

# Self-discover the PUBLIC address the coordinator dials to reach this node. On RunPod the
# container is handed RUNPOD_PUBLIC_IP + RUNPOD_TCP_PORT_<internal> (the proxy mapping for
# our :19210), so a node needs NO hand-wiring — it figures out its own reachable address.
# An explicit CIRCUIT_ADVERTISE_* still wins (e.g. a home/bare-metal box with its own
# public IP/port). So on RunPod an operator sets only CIRCUIT_CONTROL_URL (+ key); the
# advertise address is automatic.
ADV_HOST="${CIRCUIT_ADVERTISE_HOST:-${RUNPOD_PUBLIC_IP:-}}"
ADV_PORT="${CIRCUIT_ADVERTISE_PORT:-${RUNPOD_TCP_PORT_19210:-0}}"
echo "[run-mesh] advertising ${ADV_HOST:-<bind>}:${ADV_PORT:-<bind>} -> coordinator $CIRCUIT_CONTROL_URL"

# node id derives from the ed25519 key (--node-key if given, else persisted/generated at
# --node-key-file) and signs /register, so the coordinator can enforce CIRCUIT_MESH_VERIFY_SIG.
exec python3 -u -m engine.stage_worker \
  --port 19210 --model "$CIRCUIT_MODEL" --device cuda --host 0.0.0.0 --prune \
  --control-url   "$CIRCUIT_CONTROL_URL" \
  --node-key      "${CIRCUIT_NODE_KEY:-}" \
  --node-key-file "${CIRCUIT_NODE_KEY_FILE:-/workspace/node_key.hex}" \
  --capacity-layers "${CIRCUIT_CAPACITY_LAYERS:-64}" \
  --model-fp      "${CIRCUIT_MODEL_FP:-qwen2.5-32b-awq}" \
  --advertise-host "$ADV_HOST" \
  --advertise-port "$ADV_PORT" \
  --payout-wallet "${CIRCUIT_PAYOUT_WALLET:-}"
