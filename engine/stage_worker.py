"""
stage_worker.py — a pipeline stage as a standalone process.

Loads a model, holds layers [start, end), listens on a TCP socket, and serves
the coordinator: each ACTIVATION frame -> run this stage's layers (with per-
session KV) -> return the updated hidden state as a RESULT frame. KV_CTRL frames
reset or roll back a session's cache (for speculative decode).

Run:
  python3 -m engine.stage_worker --port 5001 --layers 0:12 \
      --model Qwen/Qwen2.5-0.5B-Instruct --key <64hex> [--device cpu]

The first frame that decrypts correctly authenticates the peer (shared key);
a bad key raises WireError and the connection is dropped.
"""

from __future__ import annotations

import argparse
import socket
import struct

import torch

from engine import wire
from engine.tensors import pack_activation, unpack_activation
from engine.stage import stage_for_range
from engine.kv import StageKV
from engine.model import load_model
from engine.log import make_logger

# KV_CTRL payload: [session u32][op u8][arg u32]
KVOP_RESET = 1
KVOP_TRUNCATE = 2


def _position_ids(seq_len: int, start_pos: int, device):
    return (torch.arange(seq_len, device=device) + start_pos).unsqueeze(0)


def serve(port: int, start: int, end: int, model_id: str, key: bytes,
          device: str = "cpu", host: str = "0.0.0.0",
          prune: bool = False, keep_head: bool = False):
    log = make_logger(f"stage[{start}:{end}]")
    log("INFO", "loading model", model=model_id, device=device)
    model = load_model(model_id, device=device)
    if prune:
        from engine.model import prune_to_layers
        prune_to_layers(model, start, end, keep_head=keep_head)
        log("INFO", "pruned to owned layers", keep_head=keep_head)
    stage = stage_for_range(model, start, end)
    config = model.config
    sessions: dict[int, StageKV] = {}

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(8)
    log("INFO", "listening", port=port)

    while True:
        conn, addr = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # small frames: no Nagle
        log("INFO", "peer connected", addr=f"{addr[0]}:{addr[1]}")
        try:
            _handle(conn, key, stage, config, device, sessions, log)
        except wire.WireError as e:
            log("WARN", "connection dropped", reason=str(e))
        finally:
            conn.close()


@torch.no_grad()
def _handle(conn, key, stage, config, device, sessions, log):
    while True:
        try:
            mt, payload = wire.read_frame_keyed(conn, key)
        except wire.WireError as e:
            if "closed" in str(e):
                return
            raise

        if mt == wire.ACTIVATION:
            session, pos, _flags, hidden = unpack_activation(payload)
            hidden = hidden.to(device)
            cache = sessions.get(session)
            if cache is None or pos == 0:           # pos==0 begins a new sequence
                cache = StageKV(config)
                sessions[session] = cache
            position_ids = _position_ids(hidden.shape[1], pos, device)
            out = stage.forward(hidden, position_ids, past_key_values=cache, use_cache=True)
            wire.write_frame(conn, key, wire.RESULT, pack_activation(session, pos, out.cpu()))

        elif mt == wire.KV_CTRL:
            session, op, arg = struct.unpack(">IBI", payload[:9])
            cache = sessions.get(session)
            if cache is not None:
                if op == KVOP_RESET:
                    cache.reset()
                elif op == KVOP_TRUNCATE:
                    cache.truncate_to(arg)

        elif mt == wire.PING:
            wire.write_frame(conn, key, wire.PONG, payload)

        elif mt == wire.BYE:
            return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--layers", required=True, help="START:END (e.g. 0:12)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--key", required=True, help="64-char hex cluster key")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--prune", action="store_true",
                    help="free VRAM of layers this stage doesn't own (for big models)")
    ap.add_argument("--keep-head", action="store_true",
                    help="keep embed/norm/lm_head (for the coordinator-colocated stage)")
    a = ap.parse_args()
    start, end = (int(x) for x in a.layers.split(":"))
    serve(a.port, start, end, a.model, wire.normalize_key(a.key), a.device, a.host,
          prune=a.prune, keep_head=a.keep_head)


if __name__ == "__main__":
    main()
