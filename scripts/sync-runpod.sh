#!/usr/bin/env bash
# Sync circuit-engine to the RunPod L4 and optionally run a command there.
# RunPod's direct-TCP SSH port changes on every restart, so we read the current
# port from ~/runpod.md (the "### Current SSH port" line) rather than hardcode it.
#
#   ./scripts/sync-runpod.sh                 # just sync
#   ./scripts/sync-runpod.sh test            # sync + run the test suites
#   ./scripts/sync-runpod.sh 'python3 -m tests.test_wire'   # sync + run anything
set -euo pipefail

# RunPod's proxy IP can change too (like the SSH port) — override with CIRCUIT_RUNPOD_IP; the current
# proxy IP is the default so existing use is unchanged.
HOST="root@${CIRCUIT_RUNPOD_IP:-157.157.221.29}"
KEY=/home/watchtower/.ssh/id_ed25519
REMOTE=/workspace/circuit-engine
LOCAL="$(cd "$(dirname "$0")/.." && pwd)/"

# The port sits inside a ``` code fence under "### Current SSH port", so scan a
# few lines down and match digits anywhere. `|| true` so a miss hits the guard
# below instead of tripping `set -e` silently.
PORT="$(grep -A3 'Current SSH port' /home/watchtower/runpod.md | grep -oE '[0-9]{4,6}' | head -1 || true)"
[ -n "$PORT" ] || { echo "could not read SSH port from runpod.md" >&2; exit 1; }
SSH="ssh -p $PORT -i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=8"

echo "[sync] -> $HOST:$PORT $REMOTE"
# --no-o/--no-g: RunPod FS disallows chown; -rlptz keeps perms/times without it
rsync -rlptz --delete --no-o --no-g -e "$SSH" \
  --exclude '.venv' --exclude '.git' --exclude '__pycache__' --exclude 'models' \
  "$LOCAL" "$HOST:$REMOTE/"

CMD="${1:-}"
[ "$CMD" = "test" ] && CMD='python3 -m tests.test_wire && python3 -m tests.test_tensors'
if [ -n "$CMD" ]; then
  echo "[run] $CMD"
  $SSH "$HOST" "cd $REMOTE && (python3 -c 'import cryptography' 2>/dev/null || pip install -q cryptography); $CMD"
fi
