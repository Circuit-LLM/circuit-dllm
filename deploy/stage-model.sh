#!/usr/bin/env bash
# Stage model(s) from /workspace/hf-cache (mfs) to local RAM (/dev/shm), idempotent.
#
# Why: loading the 32B by mmap'ing it directly off the mfs network volume hangs on
# random FUSE page-faults (the outage root cause). A plain sequential copy off mfs is
# reliable, and loading from RAM has no FUSE mmap at all. So: copy once, load from RAM.
set -u
DEST="${1:-/dev/shm/hf-cache}"
SRC=/workspace/hf-cache/hub
mkdir -p "$DEST/hub"
for m in models--Qwen--Qwen2.5-32B-Instruct-AWQ models--Qwen--Qwen2.5-0.5B-Instruct; do
  if [ -d "$DEST/hub/$m" ]; then
    echo "[stage-model] $m already staged at $DEST"
  else
    echo "[stage-model] staging $m -> $DEST (sequential copy from mfs)"
    cp -a "$SRC/$m" "$DEST/hub/" || { echo "[stage-model] FAILED copying $m"; exit 1; }
  fi
done
echo "[stage-model] ready at $DEST ($(du -sh "$DEST" 2>/dev/null | cut -f1))"
