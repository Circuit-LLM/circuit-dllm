#!/bin/bash
# validate-awq-dynamic.sh — end-to-end validation of the AWQ-per-node DYNAMIC chain on ONE GPU
# (docs/AWQ_PER_NODE.md productization). Runs a mesh coordinator + a contributor node as two
# processes (loopback control plane) and serves a request — exercising: publish (slice catalog +
# manifest), CIRCUIT_MESH_CATALOG coordinator (Topology from the published layout), dynamic
# register/assign, resolve_submodel (local-slice of the assigned range), serve, inference.
# Small model so it's cheap; the 72B path is identical.
#   MODEL=Qwen/Qwen2.5-7B-Instruct-AWQ LAYOUT=0:14,14:28 N=28 bash scripts/validate-awq-dynamic.sh
set -u
cd "$(dirname "$0")/.."
export HF_HOME="${HF_HOME:-/root/hf-cache}" HF_HUB_DISABLE_TELEMETRY=1
M="${MODEL:-Qwen/Qwen2.5-7B-Instruct-AWQ}"
LAYOUT="${LAYOUT:-0:14,14:28}"; N="${N:-28}"; FP="awq-val"
COORD_END="${LAYOUT#*:}"; COORD_END="${COORD_END%%,*}"      # end of the first (coord) slice
K=$(python3 -c "print('cd'*32)")

echo "== 1. publish shards locally ($LAYOUT) =="
python3 scripts/publish-awq-shards.py "$M" /root/shards --layout "$LAYOUT" 2>&1 \
  | grep -aE "slicing|manifest|staged|Error|Traceback" | tail -6
ls /root/shards 2>&1

echo "== 2. mesh coordinator (catalog $LAYOUT) =="
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
CIRCUIT_MESH=1 CIRCUIT_MESH_LAYERS="$N" CIRCUIT_MESH_FP="$FP" CIRCUIT_MESH_CATALOG="$LAYOUT" \
CIRCUIT_LOCAL_LAYERS="0:$COORD_END" CIRCUIT_COORD_SUBMODEL="/root/shards/coord-0-$COORD_END" \
CIRCUIT_MODEL="$M" CIRCUIT_KEY="$K" CIRCUIT_DEVICE=cuda CIRCUIT_API_PORT=18931 \
CIRCUIT_CONTROL_HOST=0.0.0.0 CIRCUIT_CONTROL_PORT=18932 \
  setsid python3 -u -m engine.api > /root/coordV.log 2>&1 < /dev/null &
for i in $(seq 1 50); do sleep 3; curl -sf http://127.0.0.1:18931/health >/dev/null 2>&1 && break; done
echo "coord health: $(curl -s --max-time 5 http://127.0.0.1:18931/health | head -c 220)"
grep -aE "catalog:|fewest-fattest" /root/coordV.log | tail -2

echo "== 3. contributor node (AWQ_SHARDS=local -> slice its assigned range) =="
CIRCUIT_AWQ_SHARDS=local CIRCUIT_WORK_DIR=/root/awqcache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  setsid python3 -u -m engine.stage_worker --port 19210 --model "$M" --device cuda --host 0.0.0.0 \
    --control-url http://127.0.0.1:18932 --capacity-layers "$COORD_END" --model-fp "$FP" \
    --advertise-host 127.0.0.1 --advertise-port 19210 --node-key-file /root/nodekey.hex \
    > /root/nodeV.log 2>&1 < /dev/null &
for i in $(seq 1 80); do
  sleep 4
  grep -qaE "ready \(serving\)" /root/nodeV.log 2>/dev/null && break
  grep -qaE "Traceback|register rejected" /root/nodeV.log 2>/dev/null && break
done
echo "-- node log --"; grep -aE "joined mesh|resolved AWQ|slicing locally|submodel-loaded|listening|ready|Error|Traceback|register rejected" /root/nodeV.log | tail -10
echo "coord health after: $(curl -s --max-time 5 http://127.0.0.1:18931/health | head -c 260)"

echo "== 4. inference through the dynamic AWQ mesh =="
curl -s --max-time 120 http://127.0.0.1:18931/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"x","messages":[{"role":"user","content":"In one sentence, what is a black hole?"}],"max_tokens":50,"stream":false}'
echo
echo "tracebacks: coord=$(grep -ac Traceback /root/coordV.log) node=$(grep -ac Traceback /root/nodeV.log)"
