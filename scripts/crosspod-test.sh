#!/usr/bin/env bash
# Cross-pod smoke test: coordinator on pod 1 drives a stage worker on pod 2 over
# the encrypted wire. Uses the 0.5B (already on both pods) — this proves the
# inter-pod connection + relay, not the model. Run in background.
set -uo pipefail
KEY=/home/watchtower/.ssh/id_ed25519
# RunPod proxy IP + the per-pod TCP ports are ephemeral (they change on pod restart) — override via
# env; the values below are defaults for the pods this smoke test was last run against.
HOST="${CIRCUIT_RUNPOD_IP:-157.157.221.29}"
P1="${CROSSPOD_P1:-43452}"            # pod 1 SSH
P2="${CROSSPOD_P2:-43636}"            # pod 2 SSH
STAGE_EXT="${CROSSPOD_STAGE_EXT:-43638}"     # pod 2 stage worker, external (->19210)
MODEL="Qwen/Qwen2.5-0.5B-Instruct"
CKEY=$(python3 -c "import os;print(os.urandom(32).hex())")
SSH1="ssh -p $P1 -i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=8"
SSH2="ssh -p $P2 -i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=8"
RSYNC_OPTS="-rlptz --no-o --no-g --exclude .git --exclude .venv --exclude __pycache__ --exclude models"

echo "[1] sync code to both pods"
rsync $RSYNC_OPTS -e "$SSH1" /home/watchtower/circuit-engine/ root@$HOST:/workspace/circuit-engine/ >/dev/null 2>&1
rsync $RSYNC_OPTS -e "$SSH2" /home/watchtower/circuit-engine/ root@$HOST:/workspace/circuit-engine/ >/dev/null 2>&1

echo "[2] start stage worker on pod 2 (full 0.5B, layers 0:24, listen :19210)"
# setsid + </dev/null fully detaches so the worker survives the SSH session close
$SSH2 root@$HOST "pkill -f stage_worker 2>/dev/null; sleep 1; cd /workspace/circuit-engine && setsid bash -c 'exec python3 -u -m engine.stage_worker --port 19210 --layers 0:24 --model $MODEL --key $CKEY --device cuda --host 0.0.0.0 > /tmp/stage.log 2>&1' </dev/null >/dev/null 2>&1 & echo '  launched'"

echo "[3] wait for pod 2 worker to listen"
ok=0
for i in $(seq 1 40); do
  if $SSH2 root@$HOST "grep -q listening /tmp/stage.log 2>/dev/null"; then echo "  worker listening (after ~$((i*2))s)"; ok=1; break; fi
  sleep 2
done
[ $ok = 1 ] || { echo "  worker never listened; log:"; $SSH2 root@$HOST "tail -5 /tmp/stage.log"; }

echo "[4] connectivity probe: can pod 1 reach $HOST:$STAGE_EXT ?"
$SSH1 root@$HOST "timeout 5 bash -c '</dev/tcp/$HOST/$STAGE_EXT' && echo '  TCP reachable' || echo '  TCP UNREACHABLE (hairpin NAT?)'"

echo "[5] run coordinator on pod 1 -> pod 2 stage"
$SSH1 root@$HOST "cd /workspace/circuit-engine && CIRCUIT_MODEL='$MODEL' CIRCUIT_KEY='$CKEY' CIRCUIT_STAGES='$HOST:$STAGE_EXT' CIRCUIT_DEVICE=cuda CIRCUIT_N=20 timeout 120 python3 -u tests/run_coordinator.py 2>&1 | grep -E 'CROSSPOD|coordinator:|Error|Traceback|refused|never came up'"

echo "[6] cleanup"
$SSH2 root@$HOST "pkill -f stage_worker 2>/dev/null; echo '  pod 2 worker stopped'"
echo "DONE"
