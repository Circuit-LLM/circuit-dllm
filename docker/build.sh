#!/usr/bin/env bash
# Build + publish the GPU contributor node image. Run from the circuit-dllm repo root on a
# machine with engine access (the engine source is baked in). Contributors only ever PULL.
#
#   docker/build.sh                 # build + tag :latest and :<git-sha>
#   PUSH=1 docker/build.sh          # also push to the registry
#   REGISTRY=ghcr.io/circuit-llm docker/build.sh
set -euo pipefail

REGISTRY="${REGISTRY:-ghcr.io/circuit-llm}"
IMAGE="$REGISTRY/gpu-node"
SHA="$(git rev-parse --short HEAD 2>/dev/null || echo dev)"
cd "$(git rev-parse --show-toplevel 2>/dev/null || dirname "$(dirname "$0")")"

echo "[build] $IMAGE:latest  +  :$SHA   (context: $PWD)"
docker build -f docker/Dockerfile -t "$IMAGE:latest" -t "$IMAGE:$SHA" .

if [ "${PUSH:-0}" = "1" ]; then
  echo "[build] pushing…"
  docker push "$IMAGE:latest"
  docker push "$IMAGE:$SHA"
  echo "[build] pushed $IMAGE:{latest,$SHA}"
else
  echo "[build] built locally. Set PUSH=1 to publish. Smoke test:"
  echo "        docker run --rm --gpus all -e CIRCUIT_CONTROL_URL=http://127.0.0.1:18932 $IMAGE:latest"
fi
