#!/usr/bin/env bash
# 3.3b end-to-end on ONE pod, all localhost (the real role modes as processes):
#   standalone control plane  +  3 self-registering stage workers  +  1 head-only orchestrator
#   -> POST /v1/chat/completions to the orchestrator -> it routes through the control plane to the
#      holders and streams a real completion back.  Proves the floating-coordinator SERVICE wiring.
set -uo pipefail
cd /workspace/ce
MODEL=Qwen/Qwen2.5-0.5B-Instruct
export CIRCUIT_KEY=$(python3 -c "import os;print(os.urandom(32).hex())")
LOG=/tmp/e2e; mkdir -p $LOG
pkill -f "engine.api" 2>/dev/null; pkill -f "engine.stage_worker" 2>/dev/null; sleep 1

echo "=== 1. standalone control plane (CIRCUIT_ROLE=control), verify_sig ON ==="
CIRCUIT_ROLE=control CIRCUIT_MESH=1 CIRCUIT_MESH_LAYERS=24 CIRCUIT_MESH_STAGES=3 \
  CIRCUIT_MESH_VERIFY_SIG=1 CIRCUIT_CONTROL_HOST=127.0.0.1 CIRCUIT_CONTROL_PORT=18932 \
  CIRCUIT_REAP_INTERVAL=5 \
  python3 -m engine.api > $LOG/control.log 2>&1 &
sleep 3
curl -s http://127.0.0.1:18932/health && echo

echo "=== 2. three stage workers join the control plane (control-client mode) ==="
for i in 0 1 2; do
  CIRCUIT_KEY=$CIRCUIT_KEY python3 -m engine.stage_worker --port $((19310+i)) --model $MODEL \
    --device cpu --host 127.0.0.1 --control-url http://127.0.0.1:18932 \
    --node-key-file $LOG/w$i.hex --capacity-layers 8 > $LOG/w$i.log 2>&1 &
done

echo "--- wait for coverage_ok ---"
cov=
for t in $(seq 1 50); do
  cov=$(curl -s http://127.0.0.1:18932/health 2>/dev/null | python3 -c "import sys,json;print(json.load(sys.stdin).get('coverage_ok'))" 2>/dev/null)
  [ "$cov" = "True" ] && { echo "coverage_ok after ~$((t*3))s"; break; }
  sleep 3
done
[ "$cov" = "True" ] || { echo "FAIL: mesh never reached coverage"; tail -8 $LOG/w0.log $LOG/control.log; exit 1; }

echo "=== 3. head-only orchestrator (CIRCUIT_ROLE=orchestrator) joins + serves on 18931 ==="
CIRCUIT_ROLE=orchestrator CIRCUIT_CONTROL_URL=http://127.0.0.1:18932 CIRCUIT_MODEL=$MODEL \
  CIRCUIT_DEVICE=cpu CIRCUIT_API_HOST=127.0.0.1 CIRCUIT_API_PORT=18931 \
  CIRCUIT_ORCH_ADVERTISE=127.0.0.1 CIRCUIT_NODE_KEY_FILE=$LOG/orch.hex \
  python3 -m engine.api > $LOG/orch.log 2>&1 &
role=
for t in $(seq 1 40); do
  role=$(curl -s http://127.0.0.1:18931/health 2>/dev/null | python3 -c "import sys,json;print(json.load(sys.stdin).get('role'))" 2>/dev/null)
  [ "$role" = "orchestrator" ] && { echo "orchestrator ready after ~$((t*2))s"; break; }
  sleep 2
done
[ "$role" = "orchestrator" ] || { echo "FAIL: orchestrator never ready"; tail -15 $LOG/orch.log; exit 1; }

echo "=== 4. control plane now sees orchestrator + 3 holders ==="
curl -s http://127.0.0.1:18932/topology | python3 -m json.tool
echo "--- /v1/workers on the orchestrator (head-only view) ---"
curl -s http://127.0.0.1:18931/v1/workers

echo ""
echo "=== 5. chat completion through the orchestrator (routes to the holders) ==="
resp=$(curl -s -X POST http://127.0.0.1:18931/v1/chat/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"The capital of France is\"}],\"max_tokens\":16,\"stream\":false}")
echo "$resp"
content=$(echo "$resp" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['choices'][0]['message']['content'])" 2>/dev/null)
toks=$(echo "$resp" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('usage',{}).get('completion_tokens',0))" 2>/dev/null)

echo ""
if [ -n "$content" ] && [ "${toks:-0}" -gt 0 ] 2>/dev/null; then
  echo "E2E PASSED — orchestrator served a $toks-token completion routed through the control-plane mesh:"
  echo "   \"$content\""
  rc=0
else
  echo "FAIL: no completion content (toks=$toks)"; tail -15 $LOG/orch.log; rc=1
fi

pkill -f "engine.api" 2>/dev/null; pkill -f "engine.stage_worker" 2>/dev/null
exit $rc
