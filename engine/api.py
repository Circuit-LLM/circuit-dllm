"""
api.py — OpenAI-compatible HTTP front end for the Circuit engine.

Wraps a Coordinator and exposes /v1/chat/completions (streaming + not),
/v1/models, and /health, so the distributed engine is callable by any
OpenAI client (incl. the circuitllm.xyz/dllm page). Single-stream for now:
generation is serialized behind a lock (the engine isn't concurrent yet).

Run on the coordinator pod (same env as run_coordinator.py, plus CIRCUIT_API_PORT):
  HF_HOME=/workspace/hf-cache CIRCUIT_MODEL=... CIRCUIT_KEY=... CIRCUIT_STAGES=... \
  CIRCUIT_LOCAL_LAYERS=0:32 CIRCUIT_DEVICE=cuda CIRCUIT_API_PORT=18931 \
  python3 -m engine.api
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.coordinator import Coordinator  # noqa: E402
from engine.scheduler import BatchScheduler  # noqa: E402
from engine.log import make_logger  # noqa: E402

log = make_logger("api")
_coord: Coordinator = None
_sched: BatchScheduler = None    # set when CIRCUIT_BATCH=1 (intra-step batching)


def _build_coordinator(registry=None) -> Coordinator:
    key = bytes.fromhex(os.environ["CIRCUIT_KEY"])
    # In mesh mode the stage list comes from the registry (live holders), not the
    # static CIRCUIT_STAGES env — so it's empty there. The static path is unchanged.
    stages = [(h, int(p)) for h, p in
              (a.split(":") for a in os.environ.get("CIRCUIT_STAGES", "").split(",") if a)]
    ll = os.environ.get("CIRCUIT_LOCAL_LAYERS")
    local_layers = tuple(int(x) for x in ll.split(":")) if ll else None
    return Coordinator(
        os.environ["CIRCUIT_MODEL"], stages, key,
        device=os.environ.get("CIRCUIT_DEVICE", "cuda"),
        local_layers=local_layers,
        draft_model_id=os.environ.get("CIRCUIT_DRAFT") or None,
        shard=os.environ.get("CIRCUIT_SHARD") == "1",
        other_device=os.environ.get("CIRCUIT_OTHER_DEVICE", "cpu"),
        quant=os.environ.get("CIRCUIT_QUANT", ""),
        # AWQ-per-node: a PRE-SLICED keep-head sub-model dir (embed+norm+lm_head+layers[0,k))
        # the coordinator loads whole (Marlin OK) instead of bnb-sharding. docs/AWQ_PER_NODE.md.
        submodel=os.environ.get("CIRCUIT_COORD_SUBMODEL", ""),
        registry=registry,
        # CIRCUIT_MAX_CONCURRENCY > 1 enables pipeline overlap (each request gets its
        # own stage sockets); 1 (default) = single-stream, byte-identical to before.
        max_concurrency=int(os.environ.get("CIRCUIT_MAX_CONCURRENCY", "1")),
        # CIRCUIT_CHAIN=1 forwards activations node→node (chain relay) instead of the
        # star; mesh-mode only, default off = the star path, byte-identical.
        chain_relay=os.environ.get("CIRCUIT_CHAIN") == "1",
    )


def _build_mesh():
    """Build the mesh control plane (Topology + Registry) from env — GATED on
    CIRCUIT_MESH=1. Returns (registry, host, port, reap_interval, verify_sig), or
    None when mesh mode is off (the live default → static path entirely untouched).

    The coordinator runs its co-located stage over layers [0, coordinator_end) and
    the mesh covers [coordinator_end, CIRCUIT_MESH_LAYERS) across CIRCUIT_MESH_STAGES
    slots; joined nodes register, get a slot + a derived per-node key, and serve."""
    if os.environ.get("CIRCUIT_MESH") != "1":
        return None
    from engine.topology import Topology
    from engine.registry import Registry

    ll = os.environ.get("CIRCUIT_LOCAL_LAYERS")
    coordinator_end = int(ll.split(":")[1]) if ll else 0
    layers = int(os.environ["CIRCUIT_MESH_LAYERS"])
    n_stages = int(os.environ.get("CIRCUIT_MESH_STAGES", "1"))
    slot_sizes = None
    # Catalog alignment (docs/AWQ_PER_NODE.md): if the operator points at the published shard
    # layout (CIRCUIT_MESH_CATALOG = a layout string '0:59,59:80', a manifest.json path, or a
    # repo's manifest), the coordinator builds its Topology from the SAME slot boundaries the
    # artifacts were sliced at — so every assigned slot matches a published artifact and a joining
    # node DOWNLOADS its slice instead of slicing the full checkpoint. Sets coordinator_end +
    # stage slot sizes exactly; overrides the env/fewest-fattest sizing below.
    catalog = os.environ.get("CIRCUIT_MESH_CATALOG")
    if catalog:
        from engine.shard_fetch import catalog_layout, topology_from_catalog
        coordinator_end, n_stages, slot_sizes = topology_from_catalog(layers, catalog_layout(catalog))
        print(f"[mesh] catalog: coordinator_end={coordinator_end}, {n_stages} stage slot(s) "
              f"sizes={slot_sizes} (aligned to published artifacts)", flush=True)
    else:
        # Fewest-fattest stages (SPEED_ROADMAP §1.2): if the operator declares the fleet's
        # per-node capacity (layers a node can hold, VRAM-derived), pick the FEWEST stages those
        # nodes can staff — fewer stages = fewer inter-node hops = a shorter per-token round, the
        # dominant cost on a distributed mesh. Overrides the static CIRCUIT_MESH_STAGES. Unset →
        # the static stage count (default path unchanged). Capacity-only (node-count/replication
        # is handled by rebalance as nodes join), so pass plenty of homogeneous slots.
        node_cap = os.environ.get("CIRCUIT_MESH_NODE_CAP")
        if node_cap:
            from engine.topology import plan_stages
            budget = layers - coordinator_end
            n_stages = plan_stages(budget, [int(node_cap)] * budget, replication=1)
            print(f"[mesh] fewest-fattest: {budget} layers / {node_cap}-cap nodes "
                  f"-> {n_stages} stage(s) (~{-(-budget // n_stages)} layers each), "
                  f"overriding CIRCUIT_MESH_STAGES", flush=True)
    fp = os.environ.get("CIRCUIT_MESH_FP", "")
    repl = int(os.environ.get("CIRCUIT_MESH_REPLICATION", "1"))
    dead_after = float(os.environ.get("CIRCUIT_MESH_DEAD_AFTER", "30"))
    # a distinct mesh secret is best; fall back to the cluster key for a private net
    secret = bytes.fromhex(os.environ.get("CIRCUIT_MESH_SECRET") or os.environ["CIRCUIT_KEY"])
    allow = os.environ.get("CIRCUIT_MESH_ALLOWLIST", "").strip()
    allowlist = {x.strip() for x in allow.split(",") if x.strip()} or None   # None = open (default); a set = frozen
    seed = os.environ.get("CIRCUIT_MESH_SEED_NODES", "").strip()
    seed_nodes = {x.strip() for x in seed.split(",") if x.strip()} or None    # bootstrap fleet → TRUSTED
    host = os.environ.get("CIRCUIT_CONTROL_HOST", "0.0.0.0")
    port = int(os.environ.get("CIRCUIT_CONTROL_PORT", "18932"))
    reap = float(os.environ.get("CIRCUIT_REAP_INTERVAL", "10"))
    coord_ep = (os.environ.get("CIRCUIT_COORDINATOR_ADVERTISE", host), port)

    topo = Topology(num_layers=layers, coordinator_end=coordinator_end,
                    num_stages=n_stages, model_fp=fp, replication=repl,
                    dead_after_s=dead_after,
                    slot_sizes=slot_sizes,   # set by CIRCUIT_MESH_CATALOG; None → equal/fewest-fattest
                    # topology-aware routing: coordinator's region (for region-distance
                    # estimates) + prefer-closest-holder routing. Default off → unchanged.
                    coordinator_region=os.environ.get("CIRCUIT_REGION") or None,
                    route_by_latency=os.environ.get("CIRCUIT_ROUTE_LATENCY") == "1")
    reg = Registry(topo=topo, master_secret=secret, coordinator_endpoint=coord_ep,
                   allowlist=allowlist, seed_nodes=seed_nodes)
    verify_sig = None
    if os.environ.get("CIRCUIT_MESH_VERIFY_SIG") == "1":
        from engine.control_server import make_ed25519_verifier
        verify_sig = make_ed25519_verifier()
    return reg, host, port, reap, verify_sig


def _prompt_from_messages(messages, tools=None):
    # Qwen2.5's chat template formats `tools` into the prompt and handles the
    # assistant tool_calls / role:"tool" result round-trip natively.
    return _coord.tok.apply_chat_template(
        messages, tools=tools or None, tokenize=False, add_generation_prompt=True)


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def _parse_tool_calls(text):
    """Extract Qwen-style <tool_call>{...}</tool_call> blocks into OpenAI-shaped
    tool_calls. Returns (content_or_None, tool_calls): content is whatever text
    sits outside the tool-call blocks (None when nothing meaningful remains)."""
    calls = []
    for i, m in enumerate(_TOOL_CALL_RE.finditer(text)):
        try:
            obj = json.loads(m.group(1))
        except Exception:
            continue
        name = obj.get("name")
        if not name:
            continue
        args = obj.get("arguments", {})
        # OpenAI expects function.arguments as a JSON *string*
        args_str = args if isinstance(args, str) else json.dumps(args)
        calls.append({
            "id": f"call_{int(time.time()*1000)}_{i}",
            "type": "function",
            "function": {"name": name, "arguments": args_str},
        })
    leftover = _TOOL_CALL_RE.sub("", text).strip()
    return (leftover or None), calls


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # quiet default logging
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            if _coord._dynamic:                       # mesh mode: report the live topology
                snap = _coord.registry.snapshot()
                health = {"status": "ok", "model": _coord.model_id, "mesh": True,
                          "stages": len(snap["slots"]), "coverage_ok": snap["coverage_ok"]}
            else:
                n_remote = len(_coord._stage_addrs)
                n_stages = n_remote + (1 if _coord.local_stage is not None else 0)
                health = {"status": "ok", "model": _coord.model_id, "mesh": False,
                          "stages": n_stages, "remote_stages": n_remote}
            if _coord.has_draft():                    # speculative decode active → its health
                health["speculative"] = _coord.spec_stats()
            self._json(200, health)
        elif self.path == "/v1/models":
            self._json(200, {"object": "list", "data": [
                {"id": _coord.model_id, "object": "model", "owned_by": "circuit"}]})
        elif self.path == "/v1/workers":
            # in mesh mode the holders come from the live registry, not the static list
            workers = _coord.registry.snapshot() if _coord._dynamic else _coord.stage_topology()
            self._json(200, {"workers": workers})
        else:
            self._json(404, {"error": "not found"})

    def _serve_batched(self, prompt, max_tokens, stream, tools, cid, created, model, t0):
        """Serve a request through the batch scheduler: submit it, drain its token
        queue, shape the OpenAI response (SSE or one JSON). Batched greedy — no
        speculative draft in this mode (the design trades it for throughput)."""
        out_q = _sched.submit(prompt, max_tokens)
        produced = []

        if stream:
            self.close_connection = True
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            prev_text = ""
            n = 0
            try:
                while True:
                    item = out_q.get()
                    if item is BatchScheduler.DONE:
                        break
                    if isinstance(item, Exception):
                        raise item
                    produced.append(item)
                    text = _coord.tok.decode(produced)
                    if len(text) > len(prev_text):
                        chunk = {"id": cid, "object": "chat.completion.chunk",
                                 "created": created, "model": model,
                                 "choices": [{"index": 0, "delta": {"content": text[len(prev_text):]},
                                              "finish_reason": None}]}
                        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                        self.wfile.flush()
                        prev_text = text
                        n += 1
                done = {"id": cid, "object": "chat.completion.chunk", "created": created,
                        "model": model, "choices": [{"index": 0, "delta": {},
                                                     "finish_reason": "stop"}]}
                self.wfile.write(f"data: {json.dumps(done)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                log("WARN", "client disconnected mid-stream")
            log("INFO", "done", chunks=n, secs=round(time.time() - t0, 2))
            return

        # non-streaming
        try:
            while True:
                item = out_q.get()
                if item is BatchScheduler.DONE:
                    break
                if isinstance(item, Exception):
                    raise item
                produced.append(item)
        except Exception as e:   # noqa: BLE001
            return self._json(500, {"error": str(e)})
        text = _coord.tok.decode(produced)
        message = {"role": "assistant", "content": text}
        finish = "stop"
        if tools:
            content, tool_calls = _parse_tool_calls(text)
            message = {"role": "assistant", "content": content}
            if tool_calls:
                message["tool_calls"] = tool_calls
                finish = "tool_calls"
        self._json(200, {
            "id": cid, "object": "chat.completion", "created": created, "model": model,
            "choices": [{"index": 0, "finish_reason": finish, "message": message}],
            "usage": {"completion_tokens": len(produced)},
        })
        log("INFO", "done", tokens=len(produced), secs=round(time.time() - t0, 2))

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            return self._json(404, {"error": "not found"})
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
        except Exception as e:
            return self._json(400, {"error": f"bad request: {e}"})

        messages = body.get("messages", [])
        max_tokens = int(body.get("max_tokens") or 256)
        tools = body.get("tools") or None
        # tool calls are parsed from the full generation, so they always run
        # non-stream (streaming partial tool-calls is deferred).
        stream = bool(body.get("stream", False)) and not tools
        prompt = _prompt_from_messages(messages, tools=tools)
        cid = f"chatcmpl-{int(time.time()*1000)}"
        created = int(time.time())
        model = _coord.model_id
        t0 = time.time()
        log("INFO", "request", stream=stream, max_tokens=max_tokens, msgs=len(messages), tools=bool(tools))

        if _sched is not None:   # batch mode (CIRCUIT_BATCH=1) — route through the scheduler
            return self._serve_batched(prompt, max_tokens, stream, tools, cid, created, model, t0)

        if stream:
            # un-chunked SSE has no length/chunk end-signal, so close the socket
            # when done — otherwise keep-alive leaves the client hanging after the
            # last token (never re-enabling input).
            self.close_connection = True
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            n = 0
            try:
                with _coord.request_gate():
                    # speculative decode when a draft is loaded (CIRCUIT_DRAFT),
                    # else plain greedy — same output either way, draft only speeds
                    # it up by verifying K tokens per pipeline round-trip. K is tunable
                    # via per-request "spec_k" (for sweeps with no restart) else CIRCUIT_SPEC_K:
                    # a higher K amortizes an expensive (e.g. cross-DC) round-trip over more
                    # tokens, at the cost of more wasted draft compute when acceptance is low.
                    _spec_k = int(body.get("spec_k") or os.environ.get("CIRCUIT_SPEC_K", "4"))
                    # tree drafting (verify a draft TREE per round-trip) when enabled globally
                    # (CIRCUIT_TREE=1) or per-request ("tree": true) — lets us A/B on the live
                    # mesh with no restart. Falls back to linear speculative, then plain greedy.
                    if (body.get("tree") or _coord.has_tree()) and getattr(_coord, "_draft_model", None) is not None:
                        gen = _coord.generate_tree_stream(prompt, max_tokens,
                                                          n_nodes=body.get("tree_nodes"),
                                                          branch=body.get("tree_branch"),
                                                          max_depth=body.get("tree_depth"),
                                                          beam=body.get("tree_beam"))
                    elif _coord.has_draft():
                        gen = _coord.generate_speculative_stream(prompt, max_tokens, K=_spec_k)
                    else:
                        gen = _coord.generate_stream(prompt, max_tokens)
                    for piece in gen:
                        chunk = {"id": cid, "object": "chat.completion.chunk",
                                 "created": created, "model": model,
                                 "choices": [{"index": 0, "delta": {"content": piece},
                                              "finish_reason": None}]}
                        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                        self.wfile.flush()
                        n += 1
                done = {"id": cid, "object": "chat.completion.chunk", "created": created,
                        "model": model, "choices": [{"index": 0, "delta": {},
                                                     "finish_reason": "stop"}]}
                self.wfile.write(f"data: {json.dumps(done)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                log("WARN", "client disconnected mid-stream")
            log("INFO", "done", chunks=n, secs=round(time.time() - t0, 2))
        else:
            with _coord.request_gate():
                text, toks = _coord.generate(prompt, max_tokens)
            message = {"role": "assistant", "content": text}
            finish = "stop"
            if tools:
                content, tool_calls = _parse_tool_calls(text)
                message = {"role": "assistant", "content": content}
                if tool_calls:
                    message["tool_calls"] = tool_calls
                    finish = "tool_calls"
            self._json(200, {
                "id": cid, "object": "chat.completion", "created": created, "model": model,
                "choices": [{"index": 0, "finish_reason": finish,
                             "message": message}],
                "usage": {"completion_tokens": len(toks)},
            })
            log("INFO", "done", tokens=len(toks), secs=round(time.time() - t0, 2))


def main():
    global _coord, _sched
    log("INFO", "loading engine", model=os.environ.get("CIRCUIT_MODEL"))
    mesh = _build_mesh()                       # None unless CIRCUIT_MESH=1
    if mesh:
        # Bring the control channel up BEFORE loading this coordinator's (slow) local
        # model, so a joining node registers and loads its layers IN PARALLEL with our
        # load instead of waiting for it — halves mesh cold-start downtime. Registration
        # only touches the registry/topology (no model needed); the HTTP serving path
        # below still waits for the model via _build_coordinator.
        from engine.control_server import make_server
        reg, chost, cport, reap, verify_sig = mesh
        csrv = make_server(reg, host=chost, port=cport, reap_interval=reap,
                           verify_sig=verify_sig)
        threading.Thread(target=csrv.serve_forever, daemon=True).start()
        log("INFO", "mesh control channel up — nodes may join", port=cport,
            stages=len(reg.topo.slots), replication=reg.topo.replication,
            allowlisted=(reg.allowlist is not None))

    _coord = _build_coordinator(registry=mesh[0] if mesh else None)

    # Trustless verification auditor (docs/VERIFICATION.md): periodically challenge probation
    # nodes against a trusted replica and promote/evict. OFF by default — enable with CIRCUIT_VERIFY=1
    # once validated on a real multi-node mesh. No-op without a registry or with no probation nodes,
    # and it never crashes the server (errors are logged and the loop continues).
    if mesh and os.environ.get("CIRCUIT_VERIFY") == "1":
        _audit_every = float(os.environ.get("CIRCUIT_VERIFY_INTERVAL", "30"))
        def _audit_loop():
            while True:
                time.sleep(_audit_every)
                try:
                    _coord.run_audit_round()
                except Exception as e:   # noqa: BLE001 — never let the auditor take down serving
                    log("WARN", "audit round failed", err=str(e))
        threading.Thread(target=_audit_loop, daemon=True).start()
        log("INFO", "verification auditor on", interval_s=_audit_every)

    if os.environ.get("CIRCUIT_BATCH") == "1":   # intra-step batching (Win B)
        mb = int(os.environ.get("CIRCUIT_MAX_BATCH", "8"))
        _sched = BatchScheduler(_coord, max_batch=mb)
        log("INFO", "batch scheduler on — requests are batched", max_batch=mb)

    port = int(os.environ.get("CIRCUIT_API_PORT", "18931"))
    host = os.environ.get("CIRCUIT_API_HOST", "0.0.0.0")
    srv = ThreadingHTTPServer((host, port), Handler)
    log("INFO", "API ready", port=port, model=_coord.model_id, mesh=bool(mesh))
    srv.serve_forever()


if __name__ == "__main__":
    main()
