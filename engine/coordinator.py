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
import os
import socket
import struct
import threading
import time
from collections import deque
from contextlib import contextmanager
from typing import List, Tuple

import torch

from engine import wire, chain
from engine.tensors import (pack_activation, unpack_activation, pack_batch_activation,
                            pack_tree_activation, build_tree_mask)
from engine.model import load_model, load_tokenizer, prune_to_layers
from engine.stage import stage_for_range
from engine.kv import StageKV
from engine.stage_worker import KVOP_RESET, KVOP_TRUNCATE, KVOP_FREE, KVOP_TREE_KEEP
from engine.specdecode import speculative_greedy, speculative_greedy_stream, GreedyDraft
from engine.verify import challenge_activation, outputs_agree
from engine.log import make_logger

# Throwaway session id for verification challenges — distinct from real session ids
# (itertools.count(1), small) and batch ids (>=1<<31). pos==0 resets its KV each challenge,
# so it never grows.
_AUDIT_SESSION = 1 << 30


# Failover must fail FAST: a routed mesh holder is already READY (it posted /ready
# after loading), so a healthy node connects instantly. Reaching a dead replica must
# not block on the long startup-wait below — abandon it in seconds and fail over /
# surface the coverage gap. (Startup, where a stage is still loading, keeps the long
# wait via _ensure_connected.)
_DYNAMIC_CONNECT_TIMEOUT = 8.0

# Explicit per-read timeout on stage sockets so a HUNG node (alive, accepts, never
# replies) can't block a request forever — the read raises, and the dynamic path fails
# over (the static path drops + reconnects). Generous enough not to false-positive on a
# slow node's forward; tune up for slow remote GPUs via CIRCUIT_STAGE_READ_TIMEOUT.
_STAGE_READ_TIMEOUT = float(os.environ.get("CIRCUIT_STAGE_READ_TIMEOUT", "30"))


def _connect(addr: Tuple[str, int], key: bytes, timeout: float = 180.0):
    """Connect to a stage worker, retrying until it's up (it loads a model first)."""
    host, port = addr
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            s = socket.create_connection((host, port), timeout=10)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            s.settimeout(_STAGE_READ_TIMEOUT)   # explicit read timeout (hung-node guard)
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
    __slots__ = ("socks", "conns", "pc")

    def __init__(self):
        self.socks = None
        self.conns = {}
        # Per-slot system-prompt prefix cache: this connection set's warm session +
        # the prompt token ids it currently holds. Each in-flight request owns its _Conn
        # exclusively (the pool lends one per request), so no locking is needed and the
        # cache works at any concurrency — every slot keeps its own warm prefix.
        self.pc = {"ids": None, "sid": None, "len": 0}


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
                 shard: bool = False, other_device: str = "cpu", quant: str = "",
                 submodel: str = "",
                 registry=None, max_concurrency: int = 1, chain_relay: bool = False):
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
        if submodel:
            # AWQ-per-node (docs/AWQ_PER_NODE.md): the coordinator loads a PRE-SLICED keep-head
            # sub-model = embed + norm + lm_head + layers [0,k) as a complete AWQ model (Marlin
            # OK). local_layers=(0,k); the sub-model's layers 0..k-1 ARE global [0,k) (coordinator
            # always holds the first layers, start=0), so stage_for_range maps directly.
            model = load_model(submodel, device=device)
            if local_layers is not None:
                s, e = local_layers
                self.local_stage = stage_for_range(model, s, e)
                self.log("INFO", "co-located stage (AWQ submodel)", layers=f"{s}:{e}")
        elif local_layers is not None and shard:
            # too big to load whole: load only embed/head + this pod's layers
            s, e = local_layers
            gpu = "cuda:0" if device == "cuda" else device
            if quant == "bnb":
                from engine.model import load_model_shard_bnb
                model = load_model_shard_bnb(model_id, s, e, keep_head=True, device=gpu)
                self.log("INFO", "co-located stage (bnb 4bit sharded)", layers=f"{s}:{e}")
            else:
                from engine.model import load_model_shard
                model = load_model_shard(model_id, s, e, keep_head=True,
                                         device=gpu, other_device=other_device)
                self.log("INFO", "co-located stage (sharded)", layers=f"{s}:{e}")
            self.local_stage = stage_for_range(model, s, e)
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
        # sdpa attn so tree drafting can hand the draft a 4D (non-causal) tree mask
        self._draft_model = (load_model(draft_model_id, device=device, attn_implementation="sdpa")
                             if draft_model_id else None)
        # CIRCUIT_DRAFT_KIND selects the speculative draft: "model" (default) = the proven
        # standalone GreedyDraft; "eagle" = an EAGLE head conditioned on the target's final
        # hidden (docs/EAGLE.md) — far higher acceptance, zero extra hop. Output stays
        # token-identical to greedy for any draft; the kind only changes speed.
        self._draft_kind = os.environ.get("CIRCUIT_DRAFT_KIND", "model")
        self._eagle_head = None      # lazily loaded + cached (weights shared across calls)
        self._stage_addrs = stage_addrs
        # Per-request connection isolation (pipeline overlap). The default _Conn is
        # used single-stream and for direct (un-scoped) calls; request_gate() lends a
        # pooled _Conn per concurrent request so their stage sockets never interleave.
        self._default_conn = _Conn()
        self._max_conc = max(1, int(max_concurrency))
        self._pool = _ConnPool(self._max_conc)
        self._session_ids = itertools.count(1)    # atomic id source (no lock needed)
        self._spec_lock = threading.Lock()        # guards the _spec counters/window
        # System-prompt prefix cache: the dominant TTFT cost is re-prefilling the (large,
        # identical-every-request) system prompt through the mesh. Each connection slot keeps
        # a warm session (stored on its _Conn) whose mesh+local KV holds the longest token
        # prefix shared across that slot's requests (auto-discovered by LCP); a new request
        # rolls that KV back to the shared prefix and prefills only the divergent suffix.
        # Per-slot state → works at any concurrency, lock-free (each _Conn is request-exclusive).
        self._pc_enabled = os.environ.get("CIRCUIT_PREFIX_CACHE", "1") != "0"
        self._pc_min = int(os.environ.get("CIRCUIT_PREFIX_MIN", "16"))
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
        # Chain relay (CIRCUIT_CHAIN): forward the activation node→node instead of
        # returning it to the coordinator between every stage (the star). Only meaningful
        # in dynamic/mesh mode (it needs the route's node endpoints + per-node keys).
        self._chain = bool(chain_relay) and self._dynamic
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
            # acquire_route load-balances across each slot's replicas (replication → parallel
            # pipelines); ==route_snapshot when replication=1. Pins for the session's KV affinity.
            route = self.registry.acquire_route(session)   # locked; raises on a coverage gap
            self._session_routes[session] = route
        if self._chain:                              # forward node→node (1 round-trip)
            return self._relay_chain(session, pos, hidden, route)
        for node in route:                           # star: a round-trip per stage
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

    def _relay_chain(self, session: int, pos: int, hidden: torch.Tensor, route) -> torch.Tensor:
        """Chain relay: send the activation to the HEAD holder carrying the rest of the
        route; each node computes its layers then forwards to its next hop, and the final
        result bubbles back up the chain (docs/CHAIN_RELAY.md). One coordinator round-trip
        instead of N. Output is byte-identical to the star — same nodes, same pinned order,
        same KV — only the data path changes. On a chain failure a node bubbles an ERROR
        carrying the broken hop's host:port; we suspect that node, drop the pin, and raise
        so the caller re-prefills on a fresh route (same failover contract as the star)."""
        head, downstream = chain.chain_head_and_route(route)
        key = self._node_key(head)
        try:
            sock = self._conn_for(head)
            wire.write_frame(sock, key, wire.CHAIN_ACTIVATION,
                             chain.encode_route(downstream)
                             + pack_activation(session, pos, hidden.detach().cpu()))
            mt, payload = wire.read_frame_keyed(sock, key)
        except (wire.WireError, OSError):
            self._drop_conn(head.node_id)
            self.registry.mark_suspect(head.node_id)
            self._session_routes.pop(session, None)
            raise
        if mt == wire.RESULT:
            _, _, _, hidden = unpack_activation(payload)
            return hidden.to(self.device)
        if mt == wire.ERROR:                         # a hop downstream of the head broke
            self._drop_conn(head.node_id)            # head's pooled conn may be stale now
            self._suspect_by_endpoint(payload.decode("utf-8", "replace"), route, head)
            self._session_routes.pop(session, None)
            raise wire.WireError(f"chain hop failed: {payload[:64]!r}")
        raise wire.WireError(f"chain expected RESULT/ERROR, got {wire.msg_name(mt)}")

    def _suspect_by_endpoint(self, endpoint: str, route, head) -> None:
        """Mark the route node whose `host:port` matches the failed hop SUSPECT (so the next
        route_snapshot routes around it). If NONE matches, suspect nothing — poisoning the
        healthy head (or any wrong node) on an unparseable endpoint is worse than waiting for
        the heartbeat reaper to mark the truly-dead node within dead_after_s; the caller still
        drops the session pin, so the retry re-pins on a fresh snapshot regardless."""
        for node in route:
            h, p = node.endpoint[0], int(node.endpoint[1])
            if f"{h}:{p}" == endpoint:
                self.registry.mark_suspect(node.node_id)
                return
        self.log("WARN", "chain hop-failure endpoint not in route — not suspecting any node",
                 endpoint=endpoint)

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
            if self.registry is not None:
                self.registry.release_route(session)   # free this session's replica load

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

    def _kv_truncate(self, session: int, length: int, strict: bool = False):
        """Truncate a session's KV to `length` everywhere. `strict` (used by the prefix
        cache before reusing a warm session) re-raises on any stage write failure so the
        caller can fall back to a fresh prefill rather than append onto a stale KV."""
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
                    if strict:
                        raise
            return
        for s in (conn.socks or []):
            wire.write_frame(s, self.key, wire.KV_CTRL,
                             struct.pack(">IBI", session, KVOP_TRUNCATE, length))

    # ── system-prompt prefix cache ──────────────────────────────────────────────────────
    @staticmethod
    def _lcp_len(a: list, b: list) -> int:
        n = min(len(a), len(b))
        i = 0
        while i < n and a[i] == b[i]:
            i += 1
        return i

    def _begin_session(self, ids: torch.Tensor):
        """Pick the session to decode this prompt on. With prefix caching active, reuse the
        warm session and roll its KV back to the longest prefix it shares with this prompt,
        so only the divergent suffix gets prefilled through the mesh. Returns
        (session_id, prefilled_len) — prefilled_len tokens of `ids` are already in the
        session's KV; a fresh session returns (id, 0)."""
        if not self._pc_enabled:
            return next(self._session_ids), 0
        row = ids[0].tolist()
        pc = self._active_conn().pc                  # per-slot warm session (request-exclusive)
        common = self._lcp_len(row, pc["ids"]) if pc["ids"] is not None else 0
        common = min(common, ids.shape[1] - 1)       # always leave ≥1 token to forward
        if pc["sid"] is not None and common >= self._pc_min and self._warm_route_ok(pc["sid"]):
            sid = pc["sid"]
            try:
                if common < pc["len"]:
                    self._kv_truncate(sid, common, strict=True)   # drop past the shared prefix
                pc["len"] = common
                if os.environ.get("CIRCUIT_TTFT_DEBUG"):
                    import sys as _s; print(f"[ttft] prefix HIT common={common} prompt={ids.shape[1]}", file=_s.stderr, flush=True)
                return sid, common
            except Exception as _e:                  # warm KV unusable -> rebuild fresh
                if os.environ.get("CIRCUIT_TTFT_DEBUG"):
                    import sys as _s; print(f"[ttft] prefix HIT->EXC {_e!r}", file=_s.stderr, flush=True)
                self._invalidate_prefix(pc)
        if os.environ.get("CIRCUIT_TTFT_DEBUG"):
            import sys as _s
            _have = (pc["sid"] is not None); _cm = (self._lcp_len(row, pc["ids"]) if pc["ids"] is not None else -1)
            _rok = (self._warm_route_ok(pc["sid"]) if pc["sid"] is not None else False)
            print(f"[ttft] prefix MISS prompt={ids.shape[1]} have_warm={_have} lcp={_cm} min={self._pc_min} route_ok={_rok}", file=_s.stderr, flush=True)
        return next(self._session_ids), 0

    def _warm_route_ok(self, sid: int) -> bool:
        """Reuse the warm session only when every node holding its KV is currently connected —
        otherwise we can't reliably roll that KV back and would risk appending onto a stale
        prefix. Static (non-dynamic) path has a single fixed socket set, so it's always ok."""
        if not self._dynamic:
            return True
        route = self._session_routes.get(sid)
        if not route:
            return False
        conns = self._active_conn().conns
        return all(n.node_id in conns for n in route)

    def _end_session(self, sid: int, ids: torch.Tensor):
        """Finish a generation: keep `sid` warm as the prefix cache by rolling its KV back to
        the prompt (dropping the generated tokens) and recording the prompt as the new cache
        key. On any failure (or when prefix caching is off) just free the session normally."""
        if not self._pc_enabled:
            self._reset_sessions(sid)
            return
        pc = self._active_conn().pc
        P = ids.shape[1]
        try:
            self._kv_truncate(sid, P)                # keep prompt KV, drop the generation
            old = pc.get("sid")
            if old is not None and old != sid:
                self._reset_sessions(old)            # free this slot's previous warm session
            pc["ids"], pc["sid"], pc["len"] = ids[0].tolist(), sid, P
        except Exception:
            self._invalidate_prefix(pc)
            self._reset_sessions(sid)

    def _invalidate_prefix(self, pc: dict):
        old = pc.get("sid")
        pc["ids"], pc["sid"], pc["len"] = None, None, 0
        if old is not None:
            try:
                self._reset_sessions(old)
            except Exception:
                pass

    # ── trustless verification: audit probation nodes vs a trusted replica ──────────────
    # (docs/VERIFICATION.md) Off the hot path. Challenge a probation node and a TRUSTED holder of
    # the SAME slot with the same random input, compare outputs within numerical noise, and record
    # pass/fail → promote after enough passes, evict on strikes. Lets the mesh stay open (anyone
    # joins) without a bad node ever being a primary or going unchecked.
    def _challenge_node(self, node, hidden: torch.Tensor) -> torch.Tensor:
        """Send a single-stage ACTIVATION challenge to ONE node and return its output hidden.
        Throwaway session at pos 0 (the node resets its KV on pos==0, so it never grows). Raises
        on a wire/socket error — the caller treats that as 'skip this pair', not a failed check."""
        key = self._node_key(node)
        sock = self._conn_for(node)
        wire.write_frame(sock, key, wire.ACTIVATION,
                         pack_activation(_AUDIT_SESSION, 0, hidden.detach().cpu()))
        mt, payload = wire.read_frame_keyed(sock, key)
        if mt != wire.RESULT:
            raise wire.WireError(f"audit: expected RESULT, got {wire.msg_name(mt)}")
        _, _, _, out = unpack_activation(payload)
        return out.to(self.device)

    @torch.no_grad()
    def run_audit_round(self, max_pairs: int = 4) -> dict:
        """One round of verification. For up to `max_pairs` (probation, trusted-reference) pairs on
        the same slot, challenge BOTH with the same fresh random activation and compare; record_check
        then promotes the probation node after enough passes or evicts it on strikes. Dynamic-mesh +
        registry only; a transient network error skips a pair (never counted as a failure). No-op when
        there are no probation nodes to check. Thresholds tune via CIRCUIT_VERIFY_COS / _L2."""
        if not self._dynamic or self.registry is None:
            return {"audited": 0, "results": []}
        cos_t = float(os.environ.get("CIRCUIT_VERIFY_COS", "0.99"))
        l2_t = float(os.environ.get("CIRCUIT_VERIFY_L2", "0.05"))
        width = self.config.hidden_size
        w = getattr(self.embed, "weight", None)
        dtype = w.dtype if w is not None else torch.float16
        results = []
        for test_id, ref_id, slot in self.registry.audit_pairs()[:max_pairs]:
            test = self.registry.topo.nodes.get(test_id)
            ref = self.registry.topo.nodes.get(ref_id)
            if test is None or ref is None:
                continue
            nonce = os.urandom(8).hex()              # unpredictable → a node can't precompute outputs
            x = challenge_activation(f"{test_id}:{slot}:{nonce}", width, dtype=dtype, device="cpu")
            try:
                out_ref = self._challenge_node(ref, x)
                out_test = self._challenge_node(test, x)
            except (wire.WireError, OSError):
                self._drop_conn(ref_id); self._drop_conn(test_id)
                continue                              # network blip ≠ wrong compute
            ok, cos, rel = outputs_agree(out_test, out_ref, cos_t, l2_t)
            action = self.registry.record_check(test_id, ok)
            self.log("INFO", "audit", node=test_id[:8], slot=slot, ok=ok,
                     cos=round(cos, 4), rel_l2=round(rel, 4), action=action)
            results.append({"node": test_id, "slot": slot, "ok": ok, "action": action})
        return {"audited": len(results), "results": results}

    # ── tree drafting: verify a draft TREE per round-trip (Medusa/SpecInfer-style) ──────
    # Same draft model as linear spec, but it proposes a BRANCHING tree; the mesh verifies the
    # whole tree in ONE relay (tree attention mask) and we commit the longest accepted branch.
    # Output is token-identical to greedy — the tree only changes how many tokens a single
    # network round-trip can commit. Gated by CIRCUIT_TREE=1.
    def _run_local_tree(self, session, pos, hidden, tree_positions, parents):
        cache = self._local_caches.get(session)
        if cache is None or pos == 0:
            cache = StageKV(self.config)
            self._local_caches[session] = cache
        tm = build_tree_mask(parents, pos, self.device, hidden.dtype)
        tpos = tree_positions.to(self.device).long().view(1, -1)
        return self.local_stage.forward(hidden, tpos, past_key_values=cache,
                                        use_cache=True, tree_mask=tm)

    def _relay_tree(self, session, pos, hidden, tree_positions, parents):
        """Tree verify across the mesh: local stage (tree mask) then each remote stage via
        TREE_ACTIVATION, on the session's pinned route. Returns the per-node hidden [1,T,D]."""
        if self.local_stage is not None:
            hidden = self._run_local_tree(session, pos, hidden, tree_positions, parents)
        if not self._dynamic:
            for s in self._ensure_connected():
                wire.write_frame(s, self.key, wire.TREE_ACTIVATION,
                                 pack_tree_activation(session, pos, hidden.detach().cpu(),
                                                      tree_positions, parents))
                mt, payload = wire.read_frame_keyed(s, self.key)
                if mt != wire.RESULT:
                    raise wire.WireError(f"expected RESULT, got {wire.msg_name(mt)}")
                _, _, _, hidden = unpack_activation(payload)
                hidden = hidden.to(self.device)
            return hidden
        route = self._session_routes.get(session)
        if route is None:
            route = self.registry.acquire_route(session)
            self._session_routes[session] = route
        for node in route:
            key = self._node_key(node)
            sock = self._conn_for(node)
            wire.write_frame(sock, key, wire.TREE_ACTIVATION,
                             pack_tree_activation(session, pos, hidden.detach().cpu(),
                                                  tree_positions, parents))
            mt, payload = wire.read_frame_keyed(sock, key)
            if mt != wire.RESULT:
                raise wire.WireError(f"expected RESULT, got {wire.msg_name(mt)}")
            _, _, _, hidden = unpack_activation(payload)
            hidden = hidden.to(self.device)
        return hidden

    def _kv_tree_keep(self, session, prefix_len, accepted_slots):
        """Compact every stage's KV (local + remote) to prefix + the accepted-path tree slots."""
        if session in self._local_caches:
            self._local_caches[session].keep_tree_path(prefix_len, accepted_slots)
        body = (struct.pack(">IBI", session, KVOP_TREE_KEEP, prefix_len)
                + struct.pack(">I", len(accepted_slots))
                + struct.pack(">%dI" % len(accepted_slots), *[int(s) for s in accepted_slots]))
        conn = self._active_conn()
        if self._dynamic:
            for nid, sock in list(conn.conns.items()):
                node = self.registry.topo.nodes.get(nid)
                key = self._node_key(node) if node else self.key
                try:
                    wire.write_frame(sock, key, wire.KV_CTRL, body)
                except OSError:
                    self._drop_conn(nid)
            return
        for s in (conn.socks or []):
            wire.write_frame(s, self.key, wire.KV_CTRL, body)

    @torch.no_grad()
    def _build_tree(self, base_ids, head, n_nodes, branch, max_depth):
        """Best-first draft tree rooted at `head` (node 0); node['suffix'] = tokens below head
        down to the node. Expand a node via draft(base_ids + [head] + suffix)."""
        import heapq
        draft = self._draft_model
        nodes = [{"tok": int(head), "parent": -1, "depth": 0, "suffix": [], "clp": 0.0}]
        fr = [(0.0, 0)]
        while len(nodes) < n_nodes and fr:
            _, idx = heapq.heappop(fr)
            nd = nodes[idx]
            if nd["depth"] >= max_depth:
                continue
            ctx = torch.cat([base_ids, torch.tensor([[int(head)] + nd["suffix"]],
                                                    device=self.device)], 1)
            lp = torch.log_softmax(draft(ctx).logits[0, -1].float(), -1)
            tb = torch.topk(lp, branch)
            for j in range(branch):
                if len(nodes) >= n_nodes:
                    break
                ct, clp = int(tb.indices[j]), nd["clp"] + float(tb.values[j])
                nodes.append({"tok": ct, "parent": idx, "depth": nd["depth"] + 1,
                              "suffix": nd["suffix"] + [ct], "clp": clp})
                heapq.heappush(fr, (-clp, len(nodes) - 1))
        return nodes

    @torch.no_grad()
    def _build_tree_batched(self, base_ids, head, n_nodes, branch, max_depth, beam):
        """Efficient tree-draft: prefill the prefix ONCE into the draft KV, then expand
        depth-by-depth — each depth processes the current beam (≤`beam` nodes) in ONE batched
        draft forward with a tree mask + incremental KV (no per-node re-prefill). ~depth
        forwards instead of ~n_nodes — that's the cost cut that makes tree drafting net-faster.
        Returns the same node list shape (tok/parent/depth) as the naive _build_tree."""
        from transformers import DynamicCache
        draft, dev = self._draft_model, self.device
        dt = next(draft.parameters()).dtype
        neg = torch.finfo(dt).min
        prefix = torch.cat([base_ids, torch.tensor([[int(head)]], device=dev)], 1)
        L = prefix.shape[1]
        kv = DynamicCache()
        lp = torch.log_softmax(draft(prefix, past_key_values=kv, use_cache=True).logits[0, -1].float(), -1)
        nodes = [{"tok": int(head), "parent": -1, "depth": 0}]
        # initial frontier = top-`beam` children of the root (head)
        tb = torch.topk(lp, branch)
        cand = sorted(((float(tb.values[j]), 0, int(tb.indices[j])) for j in range(branch)), reverse=True)
        frontier = []   # {node, anc:[kv slots of ancestors], clp}
        for clp, par, tok in cand[:beam]:
            nodes.append({"tok": tok, "parent": par, "depth": 1})
            frontier.append({"node": len(nodes) - 1, "anc": [], "clp": clp})
        depth = 1
        while frontier and len(nodes) < n_nodes and depth < max_depth:
            W = len(frontier)
            cache_len = kv.get_seq_length()
            toks = torch.tensor([[nodes[f["node"]]["tok"] for f in frontier]], device=dev)
            pos = torch.full((1, W), L - 1 + depth, device=dev, dtype=torch.long)
            m = torch.full((W, cache_len + W), neg, device=dev, dtype=dt)
            for k, f in enumerate(frontier):
                m[k, :L] = 0.0                      # prefix
                for s in f["anc"]:
                    m[k, s] = 0.0                   # ancestors
                m[k, cache_len + k] = 0.0           # self
            logits = draft(toks, past_key_values=kv, position_ids=pos,
                           attention_mask=m[None, None], use_cache=True).logits[0].float()  # [W,V]
            slot = [cache_len + k for k in range(W)]
            nxt = []
            for k, f in enumerate(frontier):
                tbk = torch.topk(torch.log_softmax(logits[k], -1), branch)
                nxt.extend((f["clp"] + float(tbk.values[j]), f["node"], slot[k], f["anc"], int(tbk.indices[j]))
                           for j in range(branch))
            nxt.sort(key=lambda c: -c[0])
            new_frontier = []
            for clp, par, par_slot, par_anc, tok in nxt:
                if len(nodes) >= n_nodes:
                    break
                nodes.append({"tok": tok, "parent": par, "depth": depth + 1})  # all candidates join the tree
                if len(new_frontier) < beam:                                   # top-`beam` keep expanding
                    new_frontier.append({"node": len(nodes) - 1, "anc": par_anc + [par_slot], "clp": clp})
            frontier = new_frontier
            depth += 1
        return nodes

    @staticmethod
    def _accept_tree(nodes, logits):
        """Root (head) always accepted; follow children matching the target's argmax. Returns
        (committed_after_head, accepted_node_indices_incl_head, bonus)."""
        ch = {}
        for i, n in enumerate(nodes):
            ch.setdefault(n["parent"], []).append(i)
            ch.setdefault(i, [])
        path, cur = [0], 0
        while True:
            pred = int(logits[cur].argmax())
            nx = next((c for c in ch.get(cur, []) if nodes[c]["tok"] == pred), None)
            if nx is None:
                break
            path.append(nx); cur = nx
        bonus = int(logits[cur].argmax())
        committed = [nodes[i]["tok"] for i in path[1:]] + [bonus]
        return committed, path, bonus

    def has_tree(self) -> bool:
        return self._draft_model is not None and os.environ.get("CIRCUIT_TREE") == "1"

    @torch.no_grad()
    def generate_tree_stream(self, prompt: str, n_new: int = 256,
                             n_nodes=None, branch=None, max_depth=None, beam=None):
        """Streaming tree-drafting decode. Token-identical to greedy; the tree only affects
        speed (more committed tokens per mesh round-trip). Requires a draft model."""
        if self._draft_model is None:
            raise RuntimeError("generate_tree_stream requires a draft model (set CIRCUIT_DRAFT)")
        n_nodes = n_nodes or int(os.environ.get("CIRCUIT_TREE_NODES", "48"))
        branch = branch or int(os.environ.get("CIRCUIT_TREE_BRANCH", "2"))
        max_depth = max_depth or int(os.environ.get("CIRCUIT_TREE_DEPTH", "24"))
        beam = beam or int(os.environ.get("CIRCUIT_TREE_BEAM", "4"))
        _naive = os.environ.get("CIRCUIT_TREE_NAIVE") == "1"
        sid = next(self._session_ids)
        local = {"rounds": 0, "accepted": 0, "proposed": 0, "window": []}
        try:
            ids = self.tok(prompt, return_tensors="pt").input_ids.to(self.device)
            hidden = self._relay(sid, 0, self.embed(ids))            # prefill the prompt KV
            head = int(self.lm_head(self.norm(hidden))[0, -1].argmax())
            produced, prev_text, pos = [], "", ids.shape[1]
            if head in self._eos_ids:
                return
            produced.append(head); prev_text = self.tok.decode(produced); yield prev_text
            while len(produced) < n_new:
                base = (torch.cat([ids, torch.tensor([produced[:-1]], device=self.device,
                                                     dtype=torch.long)], 1)
                        if len(produced) > 1 else ids)
                nodes = (self._build_tree(base, head, n_nodes, branch, max_depth) if _naive
                         else self._build_tree_batched(base, head, n_nodes, branch, max_depth, beam))
                flat = torch.tensor([[n["tok"] for n in nodes]], device=self.device)
                tpos = torch.tensor([pos + n["depth"] for n in nodes], device=self.device)
                parents = torch.tensor([n["parent"] for n in nodes], device=self.device)
                th = self._relay_tree(sid, pos, self.embed(flat), tpos, parents)
                logits = self.lm_head(self.norm(th))[0].float()
                committed, path, bonus = self._accept_tree(nodes, logits)
                self._kv_tree_keep(sid, pos, path)
                pos += len(path)
                local["rounds"] += 1
                local["accepted"] += len(path) - 1
                local["proposed"] += len(nodes) - 1
                head = bonus
                for t in committed:
                    if t in self._eos_ids or len(produced) >= n_new:
                        return
                    produced.append(t)
                    text = self.tok.decode(produced)
                    if len(text) > len(prev_text):
                        yield text[len(prev_text):]
                        prev_text = text
        finally:
            self._merge_spec(local)
            self._reset_sessions(sid)

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

    def has_draft(self) -> bool:
        """True when a speculative draft is available — a separate model (CIRCUIT_DRAFT) or an
        EAGLE head (CIRCUIT_DRAFT_KIND=eagle). The API uses this to choose the speculative path."""
        return self._draft_model is not None or self._draft_kind == "eagle"

    def _make_draft(self):
        """Build the speculative draft per CIRCUIT_DRAFT_KIND. 'model' (default) → the proven
        standalone GreedyDraft over the separate draft model (the co-located main model may be
        pruned and can't draft itself). 'eagle' → an EAGLE head conditioned on the target's
        final hidden (docs/EAGLE.md): far higher acceptance, zero extra hop. Output stays
        token-identical to greedy for ANY draft — the kind only affects speed."""
        if self._draft_kind == "eagle":
            from engine.eagle import EagleDraft, load_eagle_head
            if self._eagle_head is None:
                self._eagle_head = load_eagle_head(
                    os.environ["CIRCUIT_EAGLE_HEAD"], self._model, self.device)
            return EagleDraft(self._eagle_head, self._model.model.embed_tokens,
                              self._model.lm_head, self.norm, device=self.device)
        return GreedyDraft(self._draft_model or self._model, device=self.device)

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
                draft = self._make_draft()
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
        Requires a draft (the co-located main model is pruned and cannot draft itself):
        a separate model (CIRCUIT_DRAFT) or an EAGLE head (CIRCUIT_DRAFT_KIND=eagle)."""
        if self._draft_kind == "model" and self._draft_model is None:
            raise RuntimeError("generate_speculative_stream requires a draft model (set CIRCUIT_DRAFT)")
        sid = None
        ids = None
        local = {"rounds": 0, "accepted": 0, "proposed": 0, "window": []}
        try:
            ids = self.tok(prompt, return_tensors="pt").input_ids.to(self.device)
            sid, prefilled = self._begin_session(ids)   # reuse warm system-prompt KV if it matches
            target = SocketTarget(self, sid, prefilled)
            draft = self._make_draft()
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
            if sid is not None:
                self._end_session(sid, ids)   # keep the session warm as the prefix cache

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

    def __init__(self, coord: "Coordinator", session: int, prefilled: int = 0):
        self.coord = coord
        self.session = session
        self._prefilled = prefilled       # tokens of the prompt already warm in this session's KV
        self._kv = prefilled
        self._last_hidden = None

    def kv_len(self) -> int:
        return self._kv

    def last_hidden(self) -> torch.Tensor:
        """The mesh's final pre-norm hidden [1,T,D] from the most recent forward_tokens — the
        feature an EAGLE draft conditions on. Already in hand here (the coordinator applies
        norm+lm_head locally), so EAGLE adds zero extra hops."""
        return self._last_hidden

    @torch.no_grad()
    def forward_tokens(self, token_ids: torch.Tensor, start_pos: int) -> torch.Tensor:
        # Prefix cache: on the prompt prefill (start_pos==0) the session's KV already holds
        # token_ids[:self._prefilled] (the shared system-prompt prefix), so relay only the
        # divergent suffix at that position. The returned logits' last row is unchanged, which
        # is all the speculative loop reads.
        if start_pos == 0 and self._prefilled > 0:
            suffix = token_ids[:, self._prefilled:]
            hidden = self.coord._relay(self.session, self._prefilled, self.coord.embed(suffix))
            self._kv = token_ids.shape[1]
            self._last_hidden = hidden
            return self.coord.lm_head(self.coord.norm(hidden))
        hidden = self.coord.embed(token_ids)
        hidden = self.coord._relay(self.session, start_pos, hidden)
        self._kv = start_pos + token_ids.shape[1]
        self._last_hidden = hidden
        return self.coord.lm_head(self.coord.norm(hidden))

    def rollback(self, length: int) -> None:
        self.coord._kv_truncate(self.session, length)
        self._kv = length
