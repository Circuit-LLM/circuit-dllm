"""
coordinator.py — drives generation across stage workers.

Holds the embedding, final norm, and lm_head; connects to the stage workers over
the encrypted wire; runs the decode loop: embed -> relay the hidden state through
each stage in order -> norm + lm_head -> sample -> repeat. Phase 0 is greedy and
synchronous (one token at a time); async-pipelined speculative decoding is
Phase 1.
"""

from __future__ import annotations

import socket
import struct
import time
from typing import List, Tuple

import torch

from engine import wire
from engine.tensors import pack_activation, unpack_activation
from engine.model import load_model, load_tokenizer
from engine.stage_worker import KVOP_RESET, KVOP_TRUNCATE
from engine.specdecode import speculative_greedy, GreedyDraft
from engine.log import make_logger


def _connect(addr: Tuple[str, int], key: bytes, timeout: float = 60.0):
    """Connect to a stage worker, retrying until it's up (it loads a model first)."""
    host, port = addr
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            s = socket.create_connection((host, port), timeout=10)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            return s
        except OSError as e:
            last = e
            time.sleep(0.5)
    raise ConnectionError(f"stage {host}:{port} never came up: {last}")


class Coordinator:
    def __init__(self, model_id: str, stage_addrs: List[Tuple[str, int]],
                 key: bytes, device: str = "cpu"):
        self.log = make_logger("coord")
        self.key = wire.normalize_key(key)
        self.device = device
        self.log("INFO", "loading coordinator parts", model=model_id)
        model = load_model(model_id, device=device)
        self._model = model  # full model retained (its submodules are what we use)
        self.tok = load_tokenizer(model_id)
        self.embed = model.model.embed_tokens
        self.norm = model.model.norm
        self.lm_head = model.lm_head
        self.socks = []
        for addr in stage_addrs:
            self.log("INFO", "connecting stage", addr=f"{addr[0]}:{addr[1]}")
            self.socks.append(_connect(addr, self.key))
        self._session = 0

    def _relay(self, session: int, pos: int, hidden: torch.Tensor) -> torch.Tensor:
        """Send the hidden state through each stage in order; return the last output."""
        for s in self.socks:
            wire.write_frame(s, self.key, wire.ACTIVATION,
                             pack_activation(session, pos, hidden.detach().cpu()))
            mt, payload = wire.read_frame_keyed(s, self.key)
            if mt != wire.RESULT:
                raise wire.WireError(f"expected RESULT, got {wire.msg_name(mt)}")
            _, _, _, hidden = unpack_activation(payload)
            hidden = hidden.to(self.device)
        return hidden

    def _reset_sessions(self, session: int):
        for s in self.socks:
            wire.write_frame(s, self.key, wire.KV_CTRL,
                             struct.pack(">IBI", session, KVOP_RESET, 0))

    def _kv_truncate(self, session: int, length: int):
        for s in self.socks:
            wire.write_frame(s, self.key, wire.KV_CTRL,
                             struct.pack(">IBI", session, KVOP_TRUNCATE, length))

    @torch.no_grad()
    def generate(self, prompt: str, n_new: int = 20) -> Tuple[str, List[int]]:
        self._session += 1
        sid = self._session
        ids = self.tok(prompt, return_tensors="pt").input_ids.to(self.device)
        seq = ids.shape[1]

        # prefill (pos=0 tells the stages to begin a fresh sequence)
        hidden = self._relay(sid, 0, self.embed(ids))
        nxt = self.lm_head(self.norm(hidden))[:, -1].argmax(-1, keepdim=True)
        out = [int(nxt)]
        cur = seq
        for _ in range(n_new - 1):
            hidden = self._relay(sid, cur, self.embed(nxt))
            nxt = self.lm_head(self.norm(hidden))[:, -1].argmax(-1, keepdim=True)
            out.append(int(nxt))
            cur += 1
        return self.tok.decode(out), out

    @torch.no_grad()
    def generate_speculative(self, prompt: str, n_new: int = 24, K: int = 4,
                             draft=None) -> Tuple[str, List[int]]:
        """Speculative decode over the socket: local draft proposes, the split
        stages (over the wire) verify. Output == greedy for any draft."""
        self._session += 1
        sid = self._session
        ids = self.tok(prompt, return_tensors="pt").input_ids.to(self.device)
        target = SocketTarget(self, sid)
        if draft is None:
            draft = GreedyDraft(self._model, device=self.device)
        out = speculative_greedy(target, draft, ids, n_new, K=K, device=self.device)
        return self.tok.decode(out), out

    def close(self):
        for s in self.socks:
            try:
                wire.write_frame(s, self.key, wire.BYE, b"")
                s.close()
            except OSError:
                pass


class SocketTarget:
    """Target protocol for speculative_greedy backed by stage workers over the
    wire. forward_tokens embeds locally, relays the (multi-token) hidden state
    through the stages, and applies norm + lm_head locally; rollback sends a
    KV_CTRL truncate to every stage. KV length is tracked locally (the stages
    track their own; this just mirrors it for the scheduler)."""

    def __init__(self, coord: "Coordinator", session: int):
        self.coord = coord
        self.session = session
        self._kv = 0

    def kv_len(self) -> int:
        return self._kv

    @torch.no_grad()
    def forward_tokens(self, token_ids: torch.Tensor, start_pos: int) -> torch.Tensor:
        hidden = self.coord.embed(token_ids)
        hidden = self.coord._relay(self.session, start_pos, hidden)
        self._kv = start_pos + token_ids.shape[1]
        return self.coord.lm_head(self.coord.norm(hidden))

    def rollback(self, length: int) -> None:
        self.coord._kv_truncate(self.session, length)
        self._kv = length
