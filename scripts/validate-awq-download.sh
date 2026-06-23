#!/bin/bash
# validate-awq-download.sh — validate the AWQ-per-node DOWNLOAD leg (docs/AWQ_PER_NODE.md): publish
# per-stage shards to an HF repo, then a contributor node PULLS its assigned slice from that repo,
# VERIFIES its sha256 against the manifest, and serves. Proves publish→HF→download→verify→serve
# (the only leg the local-path validation didn't cover). Needs HF_TOKEN (write).
#   HF_TOKEN=... REPO=circuitllmdev/qwen25-7b-awq-shards-val bash scripts/validate-awq-download.sh
set -u
cd "$(dirname "$0")/.."
export HF_HOME="${HF_HOME:-/root/hf-cache}" HF_HUB_DISABLE_TELEMETRY=1
: "${HF_TOKEN:?set HF_TOKEN (write token)}"; export HF_TOKEN
M="${MODEL:-Qwen/Qwen2.5-7B-Instruct-AWQ}"; LAYOUT="${LAYOUT:-0:14,14:28}"; N=28; FP="awq-val"
REPO="${REPO:-circuitllmdev/qwen25-7b-awq-shards-val}"; CE=14
K=$(python3 -c "print('cd'*32)")

echo "== 1. publish + UPLOAD shards -> $REPO (slices + sha256 + revision) =="
python3 scripts/publish-awq-shards.py "$M" /root/shards --layout "$LAYOUT" --repo "$REPO" --upload 2>&1 \
  | grep -aE "slicing|hashing|manifest|uploaded|Error|Traceback" | tail -8

echo "== 2. mesh coordinator (catalog $LAYOUT; coord uses its local keep-head slice) =="
CIRCUIT_MESH=1 CIRCUIT_MESH_LAYERS=$N CIRCUIT_MESH_FP=$FP CIRCUIT_MESH_CATALOG="$LAYOUT" \
CIRCUIT_LOCAL_LAYERS="0:$CE" CIRCUIT_COORD_SUBMODEL="/root/shards/coord-0-$CE" \
CIRCUIT_MODEL="$M" CIRCUIT_KEY="$K" CIRCUIT_DEVICE=cuda CIRCUIT_API_PORT=18931 \
CIRCUIT_CONTROL_HOST=0.0.0.0 CIRCUIT_CONTROL_PORT=18932 \
  setsid python3 -u -m engine.api > /root/coordD.log 2>&1 < /dev/null &
for i in $(seq 1 50); do sleep 3; curl -sf http://127.0.0.1:18931/health >/dev/null 2>&1 && break; done
echo "coord: $(curl -s --max-time 5 http://127.0.0.1:18931/health | head -c 200)"

echo "== 3. contributor node DOWNLOADS its slice from $REPO (fresh work dir -> forces download) =="
CIRCUIT_AWQ_SHARDS="$REPO" CIRCUIT_WORK_DIR=/root/dlcache \
  setsid python3 -u -m engine.stage_worker --port 19210 --model "$M" --device cuda --host 0.0.0.0 \
    --control-url http://127.0.0.1:18932 --capacity-layers "$CE" --model-fp "$FP" \
    --advertise-host 127.0.0.1 --advertise-port 19210 --node-key-file /root/nodekey.hex \
    > /root/nodeD.log 2>&1 < /dev/null &
for i in $(seq 1 90); do
  sleep 4
  grep -qaE "ready \(serving\)" /root/nodeD.log 2>/dev/null && break
  grep -qaE "Traceback|register rejected" /root/nodeD.log 2>/dev/null && break
done
echo "-- node log (want 'downloaded published slice' + 'integrity verified', NOT 'slicing locally') --"
grep -aE "joined mesh|downloaded published|integrity verified|FAILED sha256|slicing locally|resolved AWQ|submodel-loaded|listening|ready|Error|Traceback" /root/nodeD.log | tail -12

echo "== 4. inference through the downloaded-slice mesh =="
curl -s --max-time 120 http://127.0.0.1:18931/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"x","messages":[{"role":"user","content":"In one sentence, what is gravity?"}],"max_tokens":50,"stream":false}'
echo
echo "tracebacks: coord=$(grep -ac Traceback /root/coordD.log) node=$(grep -ac Traceback /root/nodeD.log)"
