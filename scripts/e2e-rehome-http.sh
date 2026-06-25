#!/usr/bin/env bash
# HTTP-level KV-reuse RE-HOME across orchestrators (one pod, localhost):
#   control plane + 1 holder + 2 head-only orchestrators (O1, O2).
#   - reference: a single 16-token greedy completion
#   - O1 generates the first 8 tokens with circuit_keep_warm=true (its KV stays on the holder) and
#     returns {session_id, token_ids}
#   - O2 RE-HOMES that session via circuit_resume (carrying session_id + O1's token_ids): it attaches
#     to the holder's warm KV (no prompt re-prefill) and finishes
#   - assert  O1's 8  ++  O2's continuation  ==  the 16-token reference  (byte-identical re-home)
set -uo pipefail
cd /workspace/ce
MODEL=Qwen/Qwen2.5-0.5B-Instruct
export CIRCUIT_KEY=$(python3 -c "import os;print(os.urandom(32).hex())")
L=/tmp/rh; mkdir -p $L
pkill -f engine.api 2>/dev/null; pkill -f engine.stage_worker 2>/dev/null; sleep 1

CIRCUIT_ROLE=control CIRCUIT_MESH=1 CIRCUIT_MESH_LAYERS=24 CIRCUIT_MESH_STAGES=1 \
  CIRCUIT_MESH_VERIFY_SIG=1 CIRCUIT_CONTROL_HOST=127.0.0.1 CIRCUIT_CONTROL_PORT=18932 \
  CIRCUIT_REAP_INTERVAL=5 python3 -u -m engine.api > $L/control.log 2>&1 &
sleep 3
python3 -u -m engine.stage_worker --port 19320 --model $MODEL --device cpu --host 127.0.0.1 \
  --control-url http://127.0.0.1:18932 --capacity-layers 24 --hb-interval 3 \
  --node-key-file $L/h.hex > $L/h.log 2>&1 &
for t in $(seq 1 50); do
  [ "$(curl -s -m5 http://127.0.0.1:18932/health 2>/dev/null | python3 -c 'import sys,json;print(json.load(sys.stdin).get("coverage_ok"))' 2>/dev/null)" = True ] && break; sleep 3
done
for i in 0 1; do
  CIRCUIT_ROLE=orchestrator CIRCUIT_CONTROL_URL=http://127.0.0.1:18932 CIRCUIT_MODEL=$MODEL \
    CIRCUIT_DEVICE=cpu CIRCUIT_API_HOST=127.0.0.1 CIRCUIT_API_PORT=$((18941+i)) \
    CIRCUIT_ORCH_ADVERTISE=127.0.0.1 CIRCUIT_NODE_KEY_FILE=$L/o$i.hex CIRCUIT_HEARTBEAT_INTERVAL=3 \
    python3 -u -m engine.api > $L/o$i.log 2>&1 &
done
for i in 0 1; do
  for t in $(seq 1 40); do
    [ "$(curl -s -m5 http://127.0.0.1:$((18941+i))/health 2>/dev/null | python3 -c 'import sys,json;print(json.load(sys.stdin).get("role"))' 2>/dev/null)" = orchestrator ] && break; sleep 2
  done
done
echo "mesh up: control + 1 holder + O1(18941) + O2(18942)"

python3 - <<'PY'
import urllib.request, json
PROMPT = "Explain in one paragraph why the sky is blue."   # a long answer (>16 tokens, no early EOS)
def chat(port, body):
    b = json.dumps({"model":"Qwen/Qwen2.5-0.5B-Instruct",
                    "messages":[{"role":"user","content":PROMPT}], "stream":False, **body}).encode()
    r = urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
        data=b, headers={"Content-Type":"application/json"}, method="POST"), timeout=120)
    return json.load(r)

# reference: a single 16-token greedy completion on O1
ref = chat(18941, {"max_tokens":16})["circuit"]["token_ids"]

# O1: first 8 tokens, keep the holder KV warm; capture the opaque re-home blob
a = chat(18941, {"max_tokens":8, "circuit_keep_warm":True})["circuit"]
toks_a, blob = a["token_ids"], a["resume"]
assert len(toks_a) == 8, f"O1 should have produced 8 tokens, got {len(toks_a)}"

# O2: RE-HOME that session — attach to the holder's warm KV, finish (no prompt re-prefill)
b = chat(18942, {"max_tokens":8, "circuit_resume":blob})["circuit"]
combined = b["token_ids"]

print("  reference :", ref)
print("  O1(8)+O2  :", combined)
ok = combined == ref
print(f"  re-home byte-identical to reference: {ok}")
assert ok, "HTTP re-home diverged — O2 did not attach to O1's warm KV correctly"
print("HTTP RE-HOME TEST PASSED — a second orchestrator continued a session over HTTP via "
      "circuit_resume, byte-identical, with no prompt re-prefill")
PY
rc=$?
pkill -f engine.api 2>/dev/null; pkill -f engine.stage_worker 2>/dev/null
exit $rc
