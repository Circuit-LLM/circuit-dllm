#!/usr/bin/env bash
# Provision + stage the model(s) this GPU node needs. Two steps:
#
#   1. SELF-PROVISION: ensure each model is present on THIS node's own volume
#      (MODEL_HOME, default /workspace/hf-cache). If absent, download it from
#      Hugging Face — so a fresh, empty node provisions itself with no shared disk
#      and no operator copy step. This is what "hooking a new GPU to the network"
#      does: pull the weights you're responsible for, once.
#   2. STAGE TO RAM: copy from the volume (mfs) to local RAM (/dev/shm). Loading the
#      32B by mmap'ing it directly off mfs hangs on random FUSE page-faults (the old
#      outage root cause); a sequential copy off mfs is reliable and loading from RAM
#      has no FUSE mmap at all. Both steps are idempotent.
#
# Models: the 32B is always needed; the 0.5B predictive-drafting draft is only needed
# by the coordinator (CIRCUIT_DRAFT set), so a pure stage node never pulls it.
set -u
DEST="${1:-/dev/shm/hf-cache}"
MODEL_HOME="${MODEL_HOME:-/workspace/hf-cache}"
SRC="$MODEL_HOME/hub"
mkdir -p "$SRC" "$DEST/hub"

# repo_id -> on-disk cache dir name. Derived from CIRCUIT_MODEL so this script serves
# any model (32B AWQ for prod, 72B fp16 for the bnb mesh) — defaults to the 32B AWQ so
# the static prod path (which exports CIRCUIT_MODEL before calling) is unchanged.
MODEL_REPO="${CIRCUIT_MODEL:-Qwen/Qwen2.5-32B-Instruct-AWQ}"
declare -A REPOS=( ["models--${MODEL_REPO//\//--}"]="$MODEL_REPO" )
# the 0.5B predictive-drafting draft is only needed by the coordinator (CIRCUIT_DRAFT set)
[ -n "${CIRCUIT_DRAFT:-}" ] && REPOS["models--${CIRCUIT_DRAFT//\//--}"]="$CIRCUIT_DRAFT"

# 1) Ensure each model is on this node's own volume (download from HF if absent).
# "Present" must mean the WEIGHTS are actually linked in the snapshot — not just that the
# snapshot dir is non-empty. A partial/interrupted download can leave only config.json +
# model.safetensors.index.json (and even blobs without snapshot symlinks), which passes a
# naive non-empty check but then fails the load with "no model.safetensors". Re-running
# snapshot_download is idempotent: it resumes from existing blobs and (re)creates the
# missing symlinks (instant when blobs are present). So gate on a real weights file.
for m in "${!REPOS[@]}"; do
  _snap=$(ls -d "$SRC/$m/snapshots/"*/ 2>/dev/null | head -1)
  if [ -n "$_snap" ] && ls "$_snap"*.safetensors "$_snap"*.bin >/dev/null 2>&1; then
    echo "[stage-model] $m present on node volume (weights linked)"
  else
    repo="${REPOS[$m]}"
    echo "[stage-model] $m absent — downloading $repo from HF to $MODEL_HOME (one-time self-provision)"
    HF_HOME="$MODEL_HOME" python3 - "$repo" "$SRC" <<'PY' || { echo "[stage-model] DOWNLOAD FAILED for $repo"; exit 1; }
import sys
from huggingface_hub import snapshot_download
repo, cache = sys.argv[1], sys.argv[2]
p = snapshot_download(repo_id=repo, cache_dir=cache)
print("[stage-model] downloaded", repo, "->", p)
PY
  fi
done

# 2) Stage from the volume (mfs) to local RAM (/dev/shm) — sequential copy, then load from RAM.
for m in "${!REPOS[@]}"; do
  if [ -d "$DEST/hub/$m" ]; then
    echo "[stage-model] $m already in RAM ($DEST)"
  else
    echo "[stage-model] staging $m -> $DEST (sequential copy from mfs)"
    cp -a "$SRC/$m" "$DEST/hub/" || { echo "[stage-model] FAILED copying $m"; exit 1; }
  fi
done
echo "[stage-model] ready at $DEST ($(du -sh "$DEST" 2>/dev/null | cut -f1))"
