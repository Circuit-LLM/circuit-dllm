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
from engine.model import load_model, load_tokenizer, prune_to_layers
from engine.stage import stage_for_range
from engine.kv import StageKV
from engine.stage_worker import KVOP_RESET, KVOP_TRUNCATE
from engine.specdecode import speculative_greedy, speculative_greedy_stream, GreedyDraft
from engine.log import make_logger


def _connect(addr: Tuple[str, int], key: bytes, timeout: float = 180.0):
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
                 key: bytes, device: str = "cpu",
                 local_layers: Optional[Tuple[int, int]] = None,
                 draft_model_id: Optional[str] = None,
                 shard: bool = False, other_device: str = "cpu", registry=None):
        """local_layers=(s,e): run those layers IN-PROCESS (co-located stage 0)
        so a big model loads once on this pod (layers + head) instead of a
        coordinator and a stage worker each loading the whole model. The model
        is pruned to [s,e)+head; remaining layers go to the remote stage workers.
        draft_model_id: load a separate (small) draft for speculative decode —
        required when local_layers prunes the model so it can't draft itself."""
        self.log = make_logger("coord")
        self.key = wire.normalize_key(key)
        self.device = device
        self.model_id = model_id
        self._local_range = local_layers
        self.log("INFO", "loading coordinator parts", model=model_id,
                 local_layers=local_layers, shard=shard)
        self.local_stage = None
        self._local_caches = {}
        if local_layers is not None and shard:
            # too big to load whole: load only embed/head + this pod's layers
            from engine.model import load_model_shard
            s, e = local_layers
            gpu = "cuda:0" if device == "cuda" else device
            model = load_model_shard(model_id, s, e, keep_head=True,
                                     device=gpu, other_device=other_device)
            self.local_stage = stage_for_range(model, s, e)
            self.log("INFO", "co-located stage (sharded)", layers=f"{s}:{e}")
        else:
            model = load_model(model_id, device=device)
            if local_layers is not None:
                s, e = local_layers
                prune_to_layers(model, s, e, keep_head=True)
                self.local_stage = stage_for_range(model, s, e)
                self.log("INFO", "co-located stage", layers=f"{s}:{e}")
        self._model = model
        self.config = model.config
        self.tok = load_tokenizer(model_id)
        self.embed = model.model.embed_tokens
        self.norm = model.model.norm
        self.lm_head = model.lm_head
        # stop tokens: the model's generation_config eos (Qwen chat = both
        # <|im_end|> and <|endoftext|>), falling back to the tokenizer eos
        gc = getattr(model, "generation_config", None)
        eos = gc.eos_token_id if (gc and gc.eos_token_id is not None) else self.tok.eos_token_id
        self._eos_ids = set(eos) if isinstance(eos, (list, tuple)) else {eos}
        self._draft_model = load_model(draft_model_id, device=device) if draft_model_id else None
        self._stage_addrs = stage_addrs
        self.socks = None            # lazy: connect on first request, not at startup
        self._session = 0
        # Optional dynamic mesh: when a registry is attached, route via its live
        # Topology + per-node keys instead of the static stage list. The static
        # path below is left exactly as-is (backward-compatible, zero risk).
        self.registry = registry
        self._dynamic = registry is not None
        self._conns: dict = {}       # node_id -> socket (dynamic mode)

    def _ensure_connected(self):
        """Connect to remote stages on first use, so the API serves immediately
        even if a stage is still loading at startup (no startup-ordering deadlock),
        and reconnects after a drop (socks reset to None on relay error)."""
        if self.socks is None:
            socks = []
            for addr in self._stage_addrs:
                self.log("INFO", "connecting stage", addr=f"{addr[0]}:{addr[1]}")
                socks.append(_connect(addr, self.key, timeout=300.0))
            self.socks = socks
        return self.socks

    def stage_topology(self):
        """Describe the pipeline stages for /v1/workers (layerEnd inclusive)."""
        total = self.config.num_hidden_layers
        workers = []
        start = 0
        if self.local_stage is not None and self._local_range is not None:
            s, e = self._local_range
            workers.append({"nodeId": "stage0-coordinator", "layerStart": s,
                            "layerEnd": e - 1, "ready": True, "type": "gpu"})
            start = e
        n_remote = len(self._stage_addrs)
        if n_remote:
            per = max(1, (total - start) // n_remote)
            for i in range(n_remote):
                s = start + i * per
                e = total if i == n_remote - 1 else s + per
                workers.append({"nodeId": f"stage{len(workers)}-remote",
                                "layerStart": s, "layerEnd": e - 1,
                                "ready": True, "type": "gpu"})
        return workers

    def _run_local(self, session: int, pos: int, hidden: torch.Tensor) -> torch.Tensor:
        cache = self._local_caches.get(session)
        if cache is None or pos == 0:
            cache = StageKV(self.config)
            self._local_caches[session] = cache
        position_ids = (torch.arange(hidden.shape[1], device=self.device) + pos).unsqueeze(0)
        return self.local_stage.forward(hidden, position_ids, past_key_values=cache, use_cache=True)

    def _relay(self, session: int, pos: int, hidden: torch.Tensor) -> torch.Tensor:
        """Run the co-located stage (if any) then each remote stage in order."""
        if self.local_stage is not None:
            hidden = self._run_local(session, pos, hidden)
        if self._dynamic:
            return self._relay_dynamic(session, pos, hidden)
        try:
            for s in self._ensure_connected():
                wire.write_frame(s, self.key, wire.ACTIVATION,
                                 pack_activation(session, pos, hidden.detach().cpu()))
                mt, payload = wire.read_frame_keyed(s, self.key)
                if mt != wire.RESULT:
                    raise wire.WireError(f"expected RESULT, got {wire.msg_name(mt)}")
                _, _, _, hidden = unpack_activation(payload)
                hidden = hidden.to(self.device)
        except (wire.WireError, OSError):
            self.socks = None   # drop -> reconnect on the next request
            raise
        return hidden

    # ── dynamic mesh routing (used only when a registry is attached) ───────────
    def _node_key(self, node) -> bytes:
        """Per-node data-wire key issued at registration; falls back to the cluster
        key for nodes seeded without one (e.g. the static-compatible test path)."""
        k = getattr(node, "wire_key", None)
        return wire.normalize_key(k) if k else self.key

    def _conn_for(self, node):
        sock = self._conns.get(node.node_id)
        if sock is None:
            host, port = node.endpoint
            sock = _connect((host, int(port)), self._node_key(node), timeout=300.0)
            self._conns[node.node_id] = sock
        return sock

    def _drop_conn(self, node_id: str):
        sock = self._conns.pop(node_id, None)
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    def _relay_dynamic(self, session: int, pos: int, hidden: torch.Tensor) -> torch.Tensor:
        """Route through the live mesh: walk the pipeline slots in order, sending to
        each slot's primary healthy holder with that node's key. On a hop failure,
        drop the connection + mark the node suspect (Phase-2 failover will retry the
        next holder; until then the request fails and the next one re-routes)."""
        for _slot, holders in self.registry.topo.route():
            node = holders[0]
            key = self._node_key(node)
            try:
                sock = self._conn_for(node)
                wire.write_frame(sock, key, wire.ACTIVATION,
                                 pack_activation(session, pos, hidden.detach().cpu()))
                mt, payload = wire.read_frame_keyed(sock, key)
                if mt != wire.RESULT:
                    raise wire.WireError(f"expected RESULT, got {wire.msg_name(mt)}")
                _, _, _, hidden = unpack_activation(payload)
                hidden = hidden.to(self.device)
            except (wire.WireError, OSError):
                self._drop_conn(node.node_id)
                self.registry.topo.mark_suspect(node.node_id)
                raise
        return hidden

    def _reset_sessions(self, session: int):
        """Free a finished session's KV everywhere: drop the coordinator's local
        cache and tell each remote stage to free its per-session cache. Called in
        a finally on every generation path, so it must never raise — a dead socket
        just means that stage reclaims on its own restart, and a fresh session at
        pos==0 starts clean regardless."""
        self._local_caches.pop(session, None)
        if self._dynamic:
            for nid, sock in list(self._conns.items()):
                node = self.registry.topo.nodes.get(nid)
                key = self._node_key(node) if node else self.key
                try:
                    wire.write_frame(sock, key, wire.KV_CTRL,
                                     struct.pack(">IBI", session, KVOP_RESET, 0))
                except OSError:
                    self._drop_conn(nid)
            return
        for s in (self.socks or []):
            try:
                wire.write_frame(s, self.key, wire.KV_CTRL,
                                 struct.pack(">IBI", session, KVOP_RESET, 0))
            except OSError:
                self.socks = None   # drop -> reconnect on the next request
                break

    def _kv_truncate(self, session: int, length: int):
        if session in self._local_caches:
            self._local_caches[session].truncate_to(length)
        if self._dynamic:
            for nid, sock in list(self._conns.items()):
                node = self.registry.topo.nodes.get(nid)
                key = self._node_key(node) if node else self.key
                try:
                    wire.write_frame(sock, key, wire.KV_CTRL,
                                     struct.pack(">IBI", session, KVOP_TRUNCATE, length))
                except OSError:
                    self._drop_conn(nid)
            return
        for s in (self.socks or []):
            wire.write_frame(s, self.key, wire.KV_CTRL,
                             struct.pack(">IBI", session, KVOP_TRUNCATE, length))

    @torch.no_grad()
    def generate(self, prompt: str, n_new: int = 20) -> Tuple[str, List[int]]:
        self._session += 1
        sid = self._session
        try:
            ids = self.tok(prompt, return_tensors="pt").input_ids.to(self.device)
            seq = ids.shape[1]

            # prefill (pos=0 tells the stages to begin a fresh sequence)
            hidden = self._relay(sid, 0, self.embed(ids))
            nxt = self.lm_head(self.norm(hidden))[:, -1].argmax(-1, keepdim=True)
            out = []
            cur = seq
            for _ in range(n_new):
                tid = int(nxt)
                if tid in self._eos_ids:
                    break
                out.append(tid)
                hidden = self._relay(sid, cur, self.embed(nxt))
                nxt = self.lm_head(self.norm(hidden))[:, -1].argmax(-1, keepdim=True)
                cur += 1
            return self.tok.decode(out), out
        finally:
            self._reset_sessions(sid)

    @torch.no_grad()
    def generate_stream(self, prompt: str, n_new: int = 256, stop_on_eos: bool = True):
        """Greedy decode, yielding decoded text incrementally (for streaming APIs)."""
        self._session += 1
        sid = self._session
        try:
            ids = self.tok(prompt, return_tensors="pt").input_ids.to(self.device)
            cur = ids.shape[1]
            hidden = self._relay(sid, 0, self.embed(ids))
            nxt = self.lm_head(self.norm(hidden))[:, -1].argmax(-1, keepdim=True)
            produced: List[int] = []
            prev_text = ""
            for _ in range(n_new):
                tid = int(nxt)
                if stop_on_eos and tid in self._eos_ids:
                    break
                produced.append(tid)
                text = self.tok.decode(produced)
                if len(text) > len(prev_text):
                    yield text[len(prev_text):]
                    prev_text = text
                hidden = self._relay(sid, cur, self.embed(nxt))
                nxt = self.lm_head(self.norm(hidden))[:, -1].argmax(-1, keepdim=True)
                cur += 1
        finally:
            self._reset_sessions(sid)

    @torch.no_grad()
    def generate_speculative(self, prompt: str, n_new: int = 24, K: int = 4,
                             draft=None) -> Tuple[str, List[int]]:
        """Speculative decode over the socket: local draft proposes, the split
        stages (over the wire) verify. Output == greedy for any draft."""
        self._session += 1
        sid = self._session
        try:
            ids = self.tok(prompt, return_tensors="pt").input_ids.to(self.device)
            target = SocketTarget(self, sid)
            if draft is None:
                # use the separate draft model if loaded (the main model may be
                # pruned for co-located stage 0 and can't draft itself)
                draft = GreedyDraft(self._draft_model or self._model, device=self.device)
            out = speculative_greedy(target, draft, ids, n_new, K=K, device=self.device)
            return self.tok.decode(out), out
        finally:
            self._reset_sessions(sid)

    @torch.no_grad()
    def generate_speculative_stream(self, prompt: str, n_new: int = 256, K: int = 4):
        """Streaming speculative decode (yields decoded text incrementally, like
        generate_stream, but verifies K draft tokens per pipeline round-trip).
        Output is token-identical to greedy; the draft only affects speed.
        Requires a separate draft model (the co-located main model is pruned and
        cannot draft itself)."""
        if self._draft_model is None:
            raise RuntimeError("generate_speculative_stream requires a draft model (set CIRCUIT_DRAFT)")
        self._session += 1
        sid = self._session
        try:
            ids = self.tok(prompt, return_tensors="pt").input_ids.to(self.device)
            target = SocketTarget(self, sid)
            draft = GreedyDraft(self._draft_model, device=self.device)
            produced: List[int] = []
            prev_text = ""
            for tid in speculative_greedy_stream(target, draft, ids, n_new,
                                                 eos_ids=self._eos_ids, K=K,
                                                 device=self.device):
                produced.append(tid)
                text = self.tok.decode(produced)
                if len(text) > len(prev_text):
                    yield text[len(prev_text):]
                    prev_text = text
        finally:
            self._reset_sessions(sid)

    def close(self):
        if self._dynamic:
            for nid, sock in list(self._conns.items()):
                node = self.registry.topo.nodes.get(nid)
                key = self._node_key(node) if node else self.key
                try:
                    wire.write_frame(sock, key, wire.BYE, b"")
                    sock.close()
                except OSError:
                    pass
            self._conns.clear()
            return
        for s in (self.socks or []):
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
