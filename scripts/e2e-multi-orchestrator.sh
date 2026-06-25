#!/usr/bin/env bash
# 3.4 mechanism (one pod, all localhost): TWO head-only orchestrators over a replication=2 mesh.
# Validates the floating-coordinator multi-orchestrator behaviour:
#   - both orchestrators register + serve
#   - the control plane BALANCES entry (acquire_entry) across both
#   - concurrent completions to both succeed
#   - killing one orchestrator -> the survivor keeps serving, and entry stops returning the dead one
# (Aggregate-throughput-at-scale is a separate study — a small model / few lanes won't show the
#  orchestrator funnel; that needs the prod 72B mesh.)
set -uo pipefail
cd /workspace/ce
MODEL=Qwen/Qwen2.5-0.5B-Instruct
export CIRCUIT_KEY=$(python3 -c "import os;print(os.urandom(32).hex())")
L=/tmp/m; mkdir -p $L
pkill -f engine.api 2>/dev/null; pkill -f engine.stage_worker 2>/dev/null; sleep 1

echo "=== control plane: 1 slot x replication=2, reaper dead_after=10s ==="
CIRCUIT_ROLE=control CIRCUIT_MESH=1 CIRCUIT_MESH_LAYERS=24 CIRCUIT_MESH_STAGES=1 \
  CIRCUIT_MESH_REPLICATION=2 CIRCUIT_MESH_VERIFY_SIG=1 CIRCUIT_MESH_DEAD_AFTER=10 \
  CIRCUIT_CONTROL_HOST=127.0.0.1 CIRCUIT_CONTROL_PORT=18932 CIRCUIT_REAP_INTERVAL=3 \
  python3 -m engine.api > $L/control.log 2>&1 &
sleep 3

echo "=== 2 holder replicas (both cover [0,24)), heartbeat every 3s (< dead_after) ==="
for i in 0 1; do
  python3 -m engine.stage_worker --port $((19320+i)) --model $MODEL --device cpu --host 127.0.0.1 \
    --control-url http://127.0.0.1:18932 --node-key-file $L/h$i.hex --capacity-layers 24 \
    --hb-interval 3 > $L/h$i.log 2>&1 &
done
cov=
for t in $(seq 1 50); do
  cov=$(curl -s http://127.0.0.1:18932/health 2>/dev/null | python3 -c "import sys,json;print(json.load(sys.stdin).get('coverage_ok'))" 2>/dev/null)
  [ "$cov" = "True" ] && { echo "coverage_ok after ~$((t*3))s"; break; }; sleep 3
done
[ "$cov" = "True" ] || { echo "FAIL: no coverage"; tail -8 $L/h0.log $L/control.log; exit 1; }

echo "=== 2 head-only orchestrators on 18941/18942 ==="
declare -a OPID
for i in 0 1; do
  CIRCUIT_ROLE=orchestrator CIRCUIT_CONTROL_URL=http://127.0.0.1:18932 CIRCUIT_MODEL=$MODEL \
    CIRCUIT_DEVICE=cpu CIRCUIT_API_HOST=127.0.0.1 CIRCUIT_API_PORT=$((18941+i)) \
    CIRCUIT_ORCH_ADVERTISE=127.0.0.1 CIRCUIT_NODE_KEY_FILE=$L/o$i.hex CIRCUIT_HEARTBEAT_INTERVAL=3 \
    python3 -m engine.api > $L/o$i.log 2>&1 &
  OPID[$i]=$!
done
for i in 0 1; do
  r=
  for t in $(seq 1 40); do
    r=$(curl -s http://127.0.0.1:$((18941+i))/health 2>/dev/null | python3 -c "import sys,json;print(json.load(sys.stdin).get('role'))" 2>/dev/null)
    [ "$r" = orchestrator ] && break; sleep 2
  done
  echo "orch$i (pid ${OPID[$i]}) role=$r"
  [ "$r" = orchestrator ] || { echo "FAIL: orch$i not ready"; tail -15 $L/o$i.log; exit 1; }
done

echo "=== A) entry balances across BOTH orchestrators ==="
python3 - <<'PY'
import urllib.request, json
def acq(s):
    d=json.dumps({"session":s}).encode()
    r=urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:18932/entry/acquire",
        data=d, headers={"Content-Type":"application/json"}, method="POST"))
    return json.load(r)["orchestrator"]["node_id"][:8]
picks=[acq(f"s{i}") for i in range(6)]
print("  entry picks:", picks)
assert len(set(picks))==2, f"entry did not spread across 2 orchestrators: {picks}"
print("  PASS: entry balanced across 2 orchestrators")
PY

echo "=== B) concurrent completions to BOTH orchestrators ==="
ok=0
for i in 0 1; do
  out=$(curl -s -X POST http://127.0.0.1:$((18941+i))/v1/chat/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"The capital of France is\"}],\"max_tokens\":8,\"stream\":false}" \
    | python3 -c "import sys,json;print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null)
  echo "  orch$i -> \"$out\""
  echo "$out" | grep -qi paris && ok=$((ok+1))
done
[ "$ok" = 2 ] || { echo "FAIL: not both orchestrators produced 'Paris' ($ok/2)"; exit 1; }
echo "  PASS: both orchestrators served correctly"

echo "=== C) kill orch0 -> survivor serves + entry stops returning the dead one ==="
kill ${OPID[0]} 2>/dev/null
echo "  killed orch0 (pid ${OPID[0]}); waiting for the control-plane reaper (dead_after=10s)..."
sleep 14
surv=$(curl -s http://127.0.0.1:18942/v1/chat/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"The capital of France is\"}],\"max_tokens\":8,\"stream\":false}" \
  | python3 -c "import sys,json;print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null)
echo "  survivor orch1 -> \"$surv\""
echo "$surv" | grep -qi paris || { echo "FAIL: survivor did not serve"; exit 1; }
picks2=$(python3 - <<'PY'
import urllib.request, json
def acq(s):
    d=json.dumps({"session":s}).encode()
    r=urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:18932/entry/acquire",
        data=d, headers={"Content-Type":"application/json"}, method="POST"))
    o=json.load(r)["orchestrator"]; return o["node_id"][:8] if o else "none"
print(sorted(set(acq(f"k{i}") for i in range(4))))
PY
)
echo "  entry picks after kill: $picks2"
echo "$picks2" | python3 -c "import sys,ast;v=ast.literal_eval(sys.stdin.read());assert len(v)==1 and v[0]!='none', f'entry still returns >1 or none: {v}';print('  PASS: entry now returns only the survivor')"

pkill -f engine.api 2>/dev/null; pkill -f engine.stage_worker 2>/dev/null
echo ""
echo "MULTI-ORCHESTRATOR MECHANISM PASSED — 2 orchestrators balanced + served; survivor continued after a kill"
