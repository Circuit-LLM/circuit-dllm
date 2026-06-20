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
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.coordinator import Coordinator  # noqa: E402
from engine.log import make_logger  # noqa: E402

log = make_logger("api")
_coord: Coordinator = None
_lock = threading.Lock()


def _build_coordinator() -> Coordinator:
    key = bytes.fromhex(os.environ["CIRCUIT_KEY"])
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
    )


def _prompt_from_messages(messages):
    return _coord.tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)


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
            n_remote = len(_coord.socks)
            n_stages = n_remote + (1 if _coord.local_stage is not None else 0)
            self._json(200, {"status": "ok", "model": _coord.model_id,
                             "stages": n_stages, "remote_stages": n_remote})
        elif self.path == "/v1/models":
            self._json(200, {"object": "list", "data": [
                {"id": _coord.model_id, "object": "model", "owned_by": "circuit"}]})
        else:
            self._json(404, {"error": "not found"})

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
        stream = bool(body.get("stream", False))
        prompt = _prompt_from_messages(messages)
        cid = f"chatcmpl-{int(time.time()*1000)}"
        created = int(time.time())
        model = _coord.model_id
        t0 = time.time()
        log("INFO", "request", stream=stream, max_tokens=max_tokens, msgs=len(messages))

        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            n = 0
            try:
                with _lock:
                    for piece in _coord.generate_stream(prompt, max_tokens):
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
            with _lock:
                text, toks = _coord.generate(prompt, max_tokens)
            self._json(200, {
                "id": cid, "object": "chat.completion", "created": created, "model": model,
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": text}}],
                "usage": {"completion_tokens": len(toks)},
            })
            log("INFO", "done", tokens=len(toks), secs=round(time.time() - t0, 2))


def main():
    global _coord
    log("INFO", "loading engine", model=os.environ.get("CIRCUIT_MODEL"))
    _coord = _build_coordinator()
    port = int(os.environ.get("CIRCUIT_API_PORT", "18931"))
    host = os.environ.get("CIRCUIT_API_HOST", "0.0.0.0")
    srv = ThreadingHTTPServer((host, port), Handler)
    log("INFO", "API ready", port=port, model=_coord.model_id)
    srv.serve_forever()


if __name__ == "__main__":
    main()
