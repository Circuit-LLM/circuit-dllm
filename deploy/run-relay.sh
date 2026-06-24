#!/usr/bin/env bash
# Co-located NAT relay supervisor (docs/RELAY.md).
#
# Run this ALONGSIDE the coordinator on the same pod so home-desktop GPUs behind NAT can join:
# such a node dials OUT to this relay (it can't be dialed inbound), and the coordinator reaches it
# THROUGH the relay — the coordinator talks to the relay over localhost, so only ONE thing needs a
# public address: this relay's port. **The pod MUST expose CIRCUIT_RELAY_PORT (default 18940)
# publicly** so home nodes can connect to it. Cloud nodes don't need the relay at all.
#
# The relay needs no GPU (pure sockets). It reads CIRCUIT_RELAY_TOKEN from the env (empty = open
# dial; set it on BOTH the relay and the coordinator to gate who can bridge). Restarts on exit.
set -u
ENGINE_DIR="${CIRCUIT_ENGINE_DIR:-/root/circuit-engine}"
cd "$ENGINE_DIR" || exit 1
PORT="${CIRCUIT_RELAY_PORT:-18940}"
while true; do
  echo "[relay] starting on :$PORT"
  python3 -u -m engine.relay --port "$PORT" || true
  echo "[relay] exited; restarting in 2s"
  sleep 2
done
