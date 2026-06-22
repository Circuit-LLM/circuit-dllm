"""
coordinator.py — drives generation across stage workers.

Holds the embedding, final norm, and lm_head; connects to the stage workers over
the encrypted wire; runs the decode loop: embed -> relay the hidden state through
each stage in order -> norm + lm_head -> sample -> repeat. Phase 0 is greedy and
synchronous (one token at a time); async-pipelined speculative decoding is
Phase 1.
"""

from __future__ import annotations

import contextvars
import itertools
import socket
import struct
import threading
import time
from collections import deque
from contextlib import contextmanager
from typing import List, Tuple

import torch

from engine import wire
from engine.tensors import pack_activation, unpack_activation, pack_batch_activation
from engine.model import load_model, load_tokenizer, prune_to_layers
from engine.stage import stage_for_range
from engine.kv import StageKV
from engine.stage_worker import KVOP_RESET, KVOP_TRUNCATE, KVOP_FREE
from engine.specdecode import speculative_greedy, speculative_greedy_stream, GreedyDraft
from engine.log import make_logger


# Failover must fail FAST: a routed mesh holder is already READY (it posted /ready
# after loading), so a healthy node connects instantly. Reaching a dead replica must
# not block on the long startup-wait below — abandon it in seconds and fail over /
# surface the coverage gap. (Startup, where a stage is still loading, keeps the long
# wait via _ensure_connected.)
_DYNAMIC_CONNECT_TIMEOUT = 8.0


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


# ── per-request connection isolation (pipeline overlap) ───────────────────────
# Concurrent requests must not share a stage socket — the framed wire can't
# interleave. Each in-flight request runs inside a connection scope that carries its
# own _Conn (its own sockets); the relay path reads it from a contextvar, so nothing
# is threaded through the failover signatures. Single-stream (max_concurrency=1) keeps
# using the coordinator's default _Conn and is byte-identical to before.
_current_conn = contextvars.ContextVar("circuit_conn", default=None)


class _Conn:
    """One request's connection set. `socks` = the static-path stage list (lazy);
    `conns` = the dynamic-path node_id -> socket map. A coordinator uses one or the
    other depending on mode; both live here so the relay code is mode-agnostic."""
    __slots__ = ("socks", "conns")

    def __init__(self):
        self.socks = None
        self.conns = {}


class _ConnPool:
    """Bounds concurrency and lends each in-flight request its own _Conn, reused (not
    churned). Size 1 = a single connection set, serialised — i.e. today's single
    stream. `all` keeps every _Conn ever lent so close() can shut them at teardown."""

    def __init__(self, maxsize: int):
        self._sem = threading.Semaphore(max(1, maxsize))
        self._idle: List[_Conn] = []
        self._lock = threading.Lock()
        self.all: List[_Conn] = []

    @contextmanager
    def scope(self):
        self._sem.acquire()
        with self._lock:
            if self._idle:
                conn = self._idle.pop()
            else:
                conn = _Conn()
                self.all.append(conn)
        token = _current_conn.set(conn)
        try:
            yield conn
        finally:
            _current_conn.reset(token)
            with self._lock:
                self._idle.append(conn)
            self._sem.release()


class Coordinator:
    def __init__(self, model_id: str, stage_addrs: List[Tuple[str, int]],
                 key: bytes, device: str = "cpu",
                 local_layers: Optional[Tuple[int, int]] = None,
                 draft_model_id: Optional[str] = None,
                 shard: bool = False, other_device: str = "cpu", registry=None,
                 max_concurrency: int = 1):
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
        # speculative-decode counters (observability — see spec_stats()). Lifetime
        # totals + a bounded window of recent rounds so late draft drift is visible
        # (a lifetime average goes sticky). Single-stream under the API lock → plain.
        self._spec = {"calls": 0, "rounds": 0, "accepted": 0, "proposed": 0,
                      "window": deque(maxlen=400)}
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
        # Per-request connection isolation (pipeline overlap). The default _Conn is
        # used single-stream and for direct (un-scoped) calls; request_gate() lends a
        # pooled _Conn per concurrent request so their stage sockets never interleave.
        self._default_conn = _Conn()
        self._max_conc = max(1, int(max_concurrency))
        self._pool = _ConnPool(self._max_conc)
        self._session_ids = itertools.count(1)    # atomic id source (no lock needed)
        self._spec_lock = threading.Lock()        # guards the _spec counters/window
        # Win B: intra-step batching. Batch ids live in a HIGH id space (>=2^31) so
        # they never collide with session ids in a worker's shared cache dict; the
        # local co-located stage keeps a per-batch KV alongside the per-session one.
        self._batch_ids = itertools.count(1 << 31)
        self._local_batch_caches: dict = {}
        # Optional dynamic mesh: when a registry is attached, route via its live
        # Topology + per-node keys instead of the static stage list. The static
        # path below is left exactly as-is (backward-compatible, zero risk).
        self.registry = registry
        self._dynamic = registry is not None
        self._session_routes: dict = {}   # session -> pinned [node per slot] (affinity)

    def _active_conn(self) -> _Conn:
        """The connection set for the current request: a per-request pooled _Conn when
        inside request_gate() (concurrent mode), else the shared default (single-stream
        / direct calls). Keyed off a contextvar, so the relay code stays signature-clean."""
        return _current_conn.get() or self._default_conn

    def request_gate(self):
        """Wrap each API request in this: it bounds concurrency to max_concurrency and
        lends the request its own connection set. At 1 it's a single serialised slot
        with one reused connection — today's single-stream behaviour."""
        return self._pool.scope()

    def _ensure_connected(self):
        """Connect this request's connection to the remote stages on first use, so the
        API serves immediately even if a stage is still loading at startup (no
        startup-ordering deadlock), and reconnects after a drop (socks reset to None)."""
        conn = self._active_conn()
        if conn.socks is None:
            socks = []
            for addr in self._stage_addrs:
                self.log("INFO", "connecting stage", addr=f"{addr[0]}:{addr[1]}")
                socks.append(_connect(addr, self.key, timeout=300.0))
            conn.socks = socks
        return conn.socks

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
            self._active_conn().socks = None   # drop -> reconnect on the next request
            raise
        return hidden

    # ── dynamic mesh routing (used only when a registry is attached) ───────────
    def _node_key(self, node) -> bytes:
        """Per-node data-wire key issued at registration; falls back to the cluster
        key for nodes seeded without one (e.g. the static-compatible test path)."""
        k = getattr(node, "wire_key", None)
        return wire.normalize_key(k) if k else self.key

    def _conn_for(self, node):
        conns = self._active_conn().conns
        sock = conns.get(node.node_id)
        if sock is None:
            host, port = node.endpoint
            # short timeout — a routed holder is READY, so this connects instantly;
            # a dead one must fail fast so failover can move on (not hang ~300s).
            sock = _connect((host, int(port)), self._node_key(node),
                            timeout=_DYNAMIC_CONNECT_TIMEOUT)
            conns[node.node_id] = sock
        return sock

    def _drop_conn(self, node_id: str):
        sock = self._active_conn().conns.pop(node_id, None)
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    def _relay_dynamic(self, session: int, pos: int, hidden: torch.Tensor) -> torch.Tensor:
        """Route through the live mesh, PINNING the route for the session's lifetime
        (session affinity → the chosen holders keep this session's KV warm). The pin
        is taken (thread-safe snapshot) at the session's first hop and reused for
        every later token; on a hop failure we drop the dead conn, mark the holder
        SUSPECT, drop the pin, and raise — the caller (_relay_with_failover) then
        re-prefills the session onto a replica and retries (mid-session failover)."""
        route = self._session_routes.get(session)
        if route is None or pos == 0:
            route = self.registry.route_snapshot()   # locked; raises on a coverage gap
            self._session_routes[session] = route
        for node in route:
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
                self.registry.mark_suspect(node.node_id)
                self._session_routes.pop(session, None)
                raise
        return hidden

    def _reprefill(self, session: int, token_ids):
        """Rebuild a session's KV after a mid-session holder death: reset every stage's
        KV (+ the local cache + the pinned route), then re-run the whole sequence so
        far at pos=0 on a fresh route — the dead holder is now SUSPECT, so the route
        snapshot picks its replica. Wasteful (the survivors re-process) but exactly
        correct: you cannot rebuild one stage's KV without the inputs the survivors
        already consumed. token_ids = the [1, L] tokens for positions 0..L-1."""
        self._reset_sessions(session)                    # RESET KV everywhere + drop the pin
        if token_ids is not None and token_ids.shape[1] > 0:
            self._relay(session, 0, self.embed(token_ids))   # re-pins route, rebuilds KV 0..L-1

    def _relay_with_failover(self, session: int, pos: int, embed_input, prefill_ids,
                             max_failovers: int = 2):
        """Relay one hop, surviving a holder dying mid-session by failing over to a
        replica. On a wire error _relay has already dropped the dead conn, marked the
        node SUSPECT, and dropped the pin; we re-prefill the session on a fresh route
        and retry the hop. Static mode (no replicas) is a plain relay."""
        if not self._dynamic:
            return self._relay(session, pos, embed_input)
        last = None
        for attempt in range(max_failovers + 1):
            try:
                if attempt > 0:   # a prior hop failed — rebuild KV on a fresh route first
                    self.log("WARN", "stage failed mid-session, failing over", pos=pos, attempt=attempt)
                    self._reprefill(session, prefill_ids)
                return self._relay(session, pos, embed_input)
            except (wire.WireError, OSError) as e:
                last = e
        raise last

    def _reset_sessions(self, session: int):
        """Free a finished session's KV everywhere: drop the coordinator's local
        cache and tell each remote stage to free its per-session cache. Called in
        a finally on every generation path, so it must never raise — a dead socket
        just means that stage reclaims on its own restart, and a fresh session at
        pos==0 starts clean regardless."""
        self._local_caches.pop(session, None)
        conn = self._active_conn()
        if self._dynamic:
            self._session_routes.pop(session, None)
            for nid, sock in list(conn.conns.items()):
                node = self.registry.topo.nodes.get(nid)
                key = self._node_key(node) if node else self.key
                try:
                    wire.write_frame(sock, key, wire.KV_CTRL,
                                     struct.pack(">IBI", session, KVOP_RESET, 0))
                except OSError:
                    self._drop_conn(nid)
            return
        for s in (conn.socks or []):
            try:
                wire.write_frame(s, self.key, wire.KV_CTRL,
                                 struct.pack(">IBI", session, KVOP_RESET, 0))
            except OSError:
                conn.socks = None   # drop -> reconnect on the next request
                break

    def _kv_truncate(self, session: int, length: int):
        if session in self._local_caches:
            self._local_caches[session].truncate_to(length)
        conn = self._active_conn()
        if self._dynamic:
            for nid, sock in list(conn.conns.items()):
                node = self.registry.topo.nodes.get(nid)
                key = self._node_key(node) if node else self.key
                try:
                    wire.write_frame(sock, key, wire.KV_CTRL,
                                     struct.pack(">IBI", session, KVOP_TRUNCATE, length))
                except OSError:
                    self._drop_conn(nid)
            return
        for s in (conn.socks or []):
            wire.write_frame(s, self.key, wire.KV_CTRL,
                             struct.pack(">IBI", session, KVOP_TRUNCATE, length))

    # ── Win B: intra-step batching (the static batch primitive the scheduler drives) ──
    def _batch_inputs(self, prompts: List[str]):
        """Tokenize + LEFT-pad prompts into [B,P] ids + [B,P] padding mask (1=real)."""
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        enc = self.tok(prompts, return_tensors="pt", padding=True)
        return enc.input_ids.to(self.device), enc.attention_mask.to(self.device)

    def _run_local_batch(self, batch_id, pos, hidden, position_ids, attn):
        cache = self._local_batch_caches.get(batch_id)
        if cache is None or pos == 0:                  # pos==0 begins a fresh batch
            cache = StageKV(self.config)
            self._local_batch_caches[batch_id] = cache
        return self.local_stage.forward(hidden, position_ids, past_key_values=cache,
                                        use_cache=True, attention_mask=attn)

    def _relay_batch(self, batch_id, pos, hidden, position_ids, attn):
        """Batched analogue of _relay: the co-located stage (if any) then each remote
        stage, carrying per-row position_ids + the 2D mask. Static-stage path (the live
        config); the scheduler runs single-threaded over the default connection."""
        if self.local_stage is not None:
            hidden = self._run_local_batch(batch_id, pos, hidden, position_ids, attn)
        for s in self._ensure_connected():
            wire.write_frame(s, self.key, wire.BATCH_ACTIVATION,
                             pack_batch_activation(batch_id, pos, hidden.detach().cpu(),
                                                   position_ids, attn))
            mt, payload = wire.read_frame_keyed(s, self.key)
            if mt != wire.RESULT:
                raise wire.WireError(f"expected RESULT, got {wire.msg_name(mt)}")
            _, _, _, hidden = unpack_activation(payload)
            hidden = hidden.to(self.device)
        return hidden

    def _reset_batch(self, batch_id):
        """Free a finished batch's KV — the local cache and each remote stage's entry."""
        self._local_batch_caches.pop(batch_id, None)
        for s in (self._active_conn().socks or []):
            try:
                wire.write_frame(s, self.key, wire.KV_CTRL,
                                 struct.pack(">IBI", batch_id & 0xFFFFFFFF, KVOP_FREE, 0))
            except OSError:
                pass

    @torch.no_grad()
    def generate_batch_stream(self, prompts: List[str], n_new=64, stop_on_eos: bool = True):
        """Decode a fixed batch of prompts together — one batched forward per step —
        yielding (row_index, token_id) as each token is produced, until every sequence
        finishes (EOS or its own token cap). The per-row stream is token-identical to
        running that prompt through generate() alone. `n_new` is an int (same cap for
        all) or a per-row list. A finished row rides along until the batch drains
        (static batching; dynamic admit/evict is the next step)."""
        bid = next(self._batch_ids)
        B = len(prompts)
        caps = list(n_new) if isinstance(n_new, (list, tuple)) else [n_new] * B
        done = [False] * B
        count = [0] * B
        try:
            ids, attn = self._batch_inputs(prompts)
            pos = (attn.long().cumsum(-1) - 1).clamp(min=0)
            hidden = self._relay_batch(bid, 0, self.embed(ids), pos, attn)
            nxt = self.lm_head(self.norm(hidden))[:, -1].argmax(-1, keepdim=True)   # [B,1]
            lengths = attn.long().sum(-1)

            def emit():
                for b in range(B):
                    if done[b] or count[b] >= caps[b]:
                        done[b] = True
                        continue
                    t = int(nxt[b])
                    if stop_on_eos and t in self._eos_ids:
                        done[b] = True
                    else:
                        count[b] += 1
                        yield b, t
                        if count[b] >= caps[b]:
                            done[b] = True

            yield from emit()
            cur_attn = attn
            for step in range(max(caps) - 1):
                if all(done):
                    break
                ones = torch.ones(B, 1, dtype=attn.dtype, device=self.device)
                cur_attn = torch.cat([cur_attn, ones], dim=1)     # decode token is real
                hidden = self._relay_batch(bid, step + 1, self.embed(nxt),
                                           lengths.unsqueeze(1), cur_attn)
                nxt = self.lm_head(self.norm(hidden))[:, -1].argmax(-1, keepdim=True)
                lengths = lengths + 1
                yield from emit()
        finally:
            self._reset_batch(bid)

    @torch.no_grad()
    def generate_batch(self, prompts: List[str], n_new=64,
                       stop_on_eos: bool = True) -> List[List[int]]:
        """Collect generate_batch_stream into per-prompt token lists (non-streaming)."""
        outs: List[List[int]] = [[] for _ in prompts]
        for b, t in self.generate_batch_stream(prompts, n_new, stop_on_eos):
            outs[b].append(t)
        return outs

    @torch.no_grad()
    def generate(self, prompt: str, n_new: int = 20) -> Tuple[str, List[int]]:
        sid = next(self._session_ids)
        try:
            ids = self.tok(prompt, return_tensors="pt").input_ids.to(self.device)
            seq = ids.shape[1]
            seq_ids = ids                            # tokens for the positions already in KV

            # prefill (pos=0 tells the stages to begin a fresh sequence)
            hidden = self._relay_with_failover(sid, 0, self.embed(ids), seq_ids[:, :0])
            nxt = self.lm_head(self.norm(hidden))[:, -1].argmax(-1, keepdim=True)
            out = []
            cur = seq
            for _ in range(n_new):
                tid = int(nxt)
                if tid in self._eos_ids:
                    break
                out.append(tid)
                hidden = self._relay_with_failover(sid, cur, self.embed(nxt), seq_ids)
                seq_ids = torch.cat([seq_ids, nxt], dim=1)
                nxt = self.lm_head(self.norm(hidden))[:, -1].argmax(-1, keepdim=True)
                cur += 1
            return self.tok.decode(out), out
        finally:
            self._reset_sessions(sid)

    @torch.no_grad()
    def generate_stream(self, prompt: str, n_new: int = 256, stop_on_eos: bool = True):
        """Greedy decode, yielding decoded text incrementally (for streaming APIs)."""
        sid = next(self._session_ids)
        try:
            ids = self.tok(prompt, return_tensors="pt").input_ids.to(self.device)
            cur = ids.shape[1]
            seq_ids = ids                            # tokens for the positions already in KV
            hidden = self._relay_with_failover(sid, 0, self.embed(ids), seq_ids[:, :0])
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
                hidden = self._relay_with_failover(sid, cur, self.embed(nxt), seq_ids)
                seq_ids = torch.cat([seq_ids, nxt], dim=1)
                nxt = self.lm_head(self.norm(hidden))[:, -1].argmax(-1, keepdim=True)
                cur += 1
        finally:
            self._reset_sessions(sid)

    @torch.no_grad()
    def generate_speculative(self, prompt: str, n_new: int = 24, K: int = 4,
                             draft=None) -> Tuple[str, List[int]]:
        """Speculative decode over the socket: local draft proposes, the split
        stages (over the wire) verify. Output == greedy for any draft."""
        sid = next(self._session_ids)
        local = {"rounds": 0, "accepted": 0, "proposed": 0, "window": []}
        try:
            ids = self.tok(prompt, return_tensors="pt").input_ids.to(self.device)
            target = SocketTarget(self, sid)
            if draft is None:
                # use the separate draft model if loaded (the main model may be
                # pruned for co-located stage 0 and can't draft itself)
                draft = GreedyDraft(self._draft_model or self._model, device=self.device)
            out = speculative_greedy(target, draft, ids, n_new, K=K, device=self.device,
                                     stats=local)
            return self.tok.decode(out), out
        finally:
            self._merge_spec(local)
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
        sid = next(self._session_ids)
        local = {"rounds": 0, "accepted": 0, "proposed": 0, "window": []}
        try:
            ids = self.tok(prompt, return_tensors="pt").input_ids.to(self.device)
            target = SocketTarget(self, sid)
            draft = GreedyDraft(self._draft_model, device=self.device)
            produced: List[int] = []
            prev_text = ""
            for tid in speculative_greedy_stream(target, draft, ids, n_new,
                                                 eos_ids=self._eos_ids, K=K,
                                                 device=self.device, stats=local):
                produced.append(tid)
                text = self.tok.decode(produced)
                if len(text) > len(prev_text):
                    yield text[len(prev_text):]
                    prev_text = text
        finally:
            self._merge_spec(local)
            self._reset_sessions(sid)

    def _merge_spec(self, local: dict):
        """Fold one finished call's speculative stats into the cumulative counters.
        Per-call accumulation is lock-free (a local dict); merging once at the end
        under the lock keeps concurrent speculative requests consistent without
        contending on the hot per-round path."""
        with self._spec_lock:
            self._spec["calls"] += 1
            self._spec["rounds"] += local["rounds"]
            self._spec["accepted"] += local["accepted"]
            self._spec["proposed"] += local["proposed"]
            self._spec["window"].extend(local["window"])   # deque(maxlen) auto-trims

    def spec_stats(self) -> dict:
        """Speculative-decode health. `acceptance_rate` is the fraction of proposed
        draft tokens the target accepted (healthy ~0.5–0.8 for a good draft);
        `tokens_per_round` is mean tokens committed per pipeline pass (1.0 = draft
        never helps, K+1 = always perfect). The lifetime figures go sticky, so
        `recent_acceptance_rate` (over the last ≤`window` rounds) is the one that
        actually surfaces a draft degrading late — watch it collapse toward 0."""
        with self._spec_lock:
            s = self._spec
            proposed, rounds = s["proposed"], s["rounds"]
            win = list(s["window"])
            accepted, calls = s["accepted"], s["calls"]
        wk = sum(k for _, k in win)
        wa = sum(m for m, _ in win)
        return {
            "calls": calls, "rounds": rounds,
            "draft_tokens_proposed": proposed,
            "draft_tokens_accepted": accepted,
            "acceptance_rate": round(accepted / proposed, 3) if proposed else None,
            "tokens_per_round": round((accepted + rounds) / rounds, 2) if rounds else None,
            "recent_acceptance_rate": round(wa / wk, 3) if wk else None,
            "recent_rounds": len(win),
        }

    def _close_conn(self, conn: _Conn):
        if self._dynamic:
            for nid, sock in list(conn.conns.items()):
                node = self.registry.topo.nodes.get(nid)
                key = self._node_key(node) if node else self.key
                try:
                    wire.write_frame(sock, key, wire.BYE, b"")
                    sock.close()
                except OSError:
                    pass
            conn.conns.clear()
        else:
            for s in (conn.socks or []):
                try:
                    wire.write_frame(s, self.key, wire.BYE, b"")
                    s.close()
                except OSError:
                    pass
            conn.socks = None

    def close(self):
        # shut every connection set ever used: the default plus all pooled ones
        seen = set()
        for conn in [self._default_conn, *self._pool.all]:
            if id(conn) not in seen:
                seen.add(id(conn))
                self._close_conn(conn)


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
