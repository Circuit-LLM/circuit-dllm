#!/usr/bin/env bash
# run-node.sh (role=node) — the unified-node boot path: start the NODE-CLIENT, which
# supervises the engine mesh worker. This is what a normal GPU operator's container runs
# (pre_start launches it when CIRCUIT_ROLE=node). The node-client (gpu-node.js) detects
# the GPU and joins the coordinator's mesh via deploy/run-mesh.sh — self-provisioning the
# model and getting a layer slice. The operator sets only CIRCUIT_CONTROL_URL (+ a volume).
NODE_CLIENT_DIR="${CIRCUIT_NODE_CLIENT_DIR:-/opt/circuit-node-client}"
export CIRCUIT_ENGINE_DIR="${CIRCUIT_ENGINE_DIR:-/opt/circuit-engine}"
exec node "$NODE_CLIENT_DIR/gpu-node.js"
