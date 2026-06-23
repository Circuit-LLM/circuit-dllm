#!/bin/bash
# Socket smoke test for the AWQ-per-node deploy paths (docs/AWQ_PER_NODE.md): bring up a
# coordinator (keep-head sub-model [0,K)) + one stage worker (sub-model [K,N)) as two processes
# on one GPU and serve a request — exercises serve(submodel=)/Coordinator(submodel=)/api env.
# Expects the two sub-model dirs already sliced (e.g. by tests/test_submodel_split_gpu.py).
#   COORD_SUB=/tmp/sub-coord-0-14 STAGE_SUB=/tmp/sub-stage-14-28 K=14 N=28 \
#     MODEL=Qwen/Qwen2.5-7B-Instruct-AWQ bash scripts/smoke-submodel.sh
set -u
cd "$(dirname "$0")/.."
export HF_HOME="${HF_HOME:-/root/hf}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct-AWQ}"
COORD_SUB="${COORD_SUB:-/tmp/sub-coord-0-14}"; STAGE_SUB="${STAGE_SUB:-/tmp/sub-stage-14-28}"
K="${K:-14}"; N="${N:-28}"; PORT="${PORT:-18931}"; SPORT="${SPORT:-19210}"
KEY=$(python3 -c "print('ab'*32)")

echo "== stage [$K,$N) sub-model =="
python3 -u -m engine.stage_worker --port "$SPORT" --model "$MODEL" --device cuda \
  --layers "$K:$N" --key "$KEY" --submodel "$STAGE_SUB" > /root/stage.log 2>&1 &
SPID=$!
for i in $(seq 1 40); do sleep 3; grep -q "listening" /root/stage.log 2>/dev/null && break; done
grep -aE 'submodel-loaded|listening|Error|Traceback' /root/stage.log | tail -2

echo "== coordinator/api (keep-head sub-model [0,$K)) =="
CIRCUIT_MODEL="$MODEL" CIRCUIT_COORD_SUBMODEL="$COORD_SUB" CIRCUIT_LOCAL_LAYERS="0:$K" \
  CIRCUIT_STAGES="127.0.0.1:$SPORT" CIRCUIT_KEY="$KEY" CIRCUIT_DEVICE=cuda CIRCUIT_API_PORT="$PORT" \
  python3 -u -m engine.api > /root/api.log 2>&1 &
APID=$!
ok=0
for i in $(seq 1 50); do sleep 3; curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ok=1; break; }; done
echo "health: $(curl -s --max-time 5 http://127.0.0.1:$PORT/health)"
if [ "$ok" = 1 ]; then
  echo "== request =="
  curl -s --max-time 90 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"x","messages":[{"role":"user","content":"Explain photosynthesis in one sentence."}],"max_tokens":60,"stream":false}'
  echo
else
  echo "API never became healthy"
fi
echo "api errs: $(grep -aE 'Error|Traceback|Exception' /root/api.log | tail -3)"
kill "$SPID" "$APID" 2>/dev/null
