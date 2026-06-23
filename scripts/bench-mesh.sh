#!/usr/bin/env bash
# bench-mesh.sh <coordinator-base-url> [model]
#
# Reproducible single-stream + aggregate-concurrency tok/s through the mesh, plus the
# speculative acceptance from /health. Use it for the honest A/B: run once per config
# (star vs CIRCUIT_CHAIN=1 vs CIRCUIT_DRAFT_KIND=eagle) and compare. Streaming engages the
# speculative path (the non-stream endpoint uses plain generate()). See docs/MESH_SESSION_RUNBOOK.md.
#
#   scripts/bench-mesh.sh http://1.2.3.4:11251
#   MAXTOK=120 scripts/bench-mesh.sh http://1.2.3.4:11251 Qwen/Qwen2.5-72B-Instruct
set -u

BASE="${1:?usage: bench-mesh.sh http://host:port [model]}"
MODEL="${2:-Qwen/Qwen2.5-72B-Instruct}"
MAXTOK="${MAXTOK:-80}"
CONC="${CONC:-1 4 8}"            # concurrency levels to sweep

PROMPTS=(
  "Explain in 4 sentences why the sky is blue."
  "Describe how a CPU cache works in 4 sentences."
  "What is photosynthesis in 4 sentences."
  "Explain TCP vs UDP in 4 sentences."
  "Summarize the theory of relativity in 4 sentences."
  "How does HTTPS keep data secure, in 4 sentences."
  "What causes ocean tides, in 4 sentences."
  "Explain how vaccines work in 4 sentences."
)

stream_tokens() {  # $1 = prompt -> prints number of streamed chunks (tokens)
  curl -s --max-time 240 -N -X POST "$BASE/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL\",\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"$1\"}],\"max_tokens\":$MAXTOK,\"temperature\":0}" \
    2>/dev/null | grep -c "chat.completion.chunk"
}

health_spec() {
  curl -s --max-time 10 "$BASE/health" 2>/dev/null | python3 -c '
import sys, json
try:
    s = json.load(sys.stdin).get("speculative") or {}
    print("accept=%s tok/round=%s rounds=%s" % (
        s.get("acceptance_rate"), s.get("tokens_per_round"), s.get("rounds")))
except Exception:
    print("(no /health speculative)")'
}

bench() {  # $1 = concurrency
  local N="$1" tmp start end secs tot=0 i
  tmp=$(mktemp -d)
  start=$(date +%s.%N)
  for i in $(seq 0 $((N - 1))); do
    stream_tokens "${PROMPTS[$((i % ${#PROMPTS[@]}))]}" > "$tmp/$i" &
  done
  wait
  end=$(date +%s.%N)
  secs=$(echo "$end - $start" | bc)
  for i in $(seq 0 $((N - 1))); do tot=$((tot + $(cat "$tmp/$i"))); done
  rm -rf "$tmp"
  printf "concurrency=%-2s  tokens=%-4s  wall=%6.1fs  AGGREGATE=%s tok/s\n" \
    "$N" "$tot" "$secs" "$(echo "scale=2; $tot / $secs" | bc)"
}

echo "== mesh benchmark: $BASE  (model=$MODEL, max_tokens=$MAXTOK) =="
echo "health (before): $(health_spec)"
for c in $CONC; do bench "$c"; done
echo "health (after):  $(health_spec)"
