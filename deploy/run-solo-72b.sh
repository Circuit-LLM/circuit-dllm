#!/usr/bin/env bash
# Whole 72B on ONE fat GPU — no mesh, no network hops. The simplest "fast-enough" deployment:
# an 80GB card (A100) holds the 4-bit 72B (~40GB) + the draft + KV. bnb quantize-at-load of all
# 80 layers onto the single GPU; the coordinator runs embed + ALL layers + lm_head locally and
# serves the OpenAI API directly (no CIRCUIT_MESH, no CIRCUIT_STAGES → single process). This is
# the compute-bound ceiling for the model on the card, with the network entirely out of the way.
set -u
ENGINE_DIR="${CIRCUIT_ENGINE_DIR:-/opt/circuit-engine}"
cd "$ENGINE_DIR" || exit 1

export CIRCUIT_MODEL="${CIRCUIT_MODEL:-Qwen/Qwen2.5-72B-Instruct}"   # fp16 source; bnb quantizes at load
export CIRCUIT_DRAFT="${CIRCUIT_DRAFT:-Qwen/Qwen2.5-1.5B-Instruct}"  # the better draft (higher acceptance)
export MODEL_HOME="${MODEL_HOME:-/root/hf-cache}"
bash "$ENGINE_DIR/deploy/stage-model.sh" "$MODEL_HOME" || exit 1
export HF_HOME="$MODEL_HOME" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1

export CIRCUIT_KEY="${CIRCUIT_KEY:-$(openssl rand -hex 32)}"
export CIRCUIT_DEVICE=cuda
export CIRCUIT_API_PORT="${CIRCUIT_API_PORT:-18931}"
export CIRCUIT_MAX_CONCURRENCY="${CIRCUIT_MAX_CONCURRENCY:-4}"
export CIRCUIT_QUANT=bnb
export CIRCUIT_SHARD=1
export CIRCUIT_LOCAL_LAYERS="${CIRCUIT_LOCAL_LAYERS:-0:80}"   # Qwen2.5-72B = 80 layers, all local
# deliberately NO CIRCUIT_MESH / CIRCUIT_STAGES → single-process, whole model, zero hops

exec python3 -u -m engine.api
