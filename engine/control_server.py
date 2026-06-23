"""
control_server.py — the node-facing control channel (HTTP/JSON) for the mesh.

Runs alongside the inference API on the coordinator, backed by a Registry. Nodes:

  POST /register  {node_id, endpoint:[host,port], capacity_layers, model_fp,
                   reachability, payout_wallet, [ts, sig]}
        -> 200 {assignment:{start,end}, session_key, model_fp, coordinator, replication}
           401 (bad signature) / 403 (not allowlisted) / 409 (model mismatch | no
           capacity) / 400 (bad body)
  POST /ready     {node_id}     -> mark a JOINING node READY (its weights are loaded)
  POST /heartbeat {node_id}     -> liveness
  POST /drain     {node_id}     -> leaving cleanly
  GET  /topology                -> live slots / holders / states (ops)
  GET  /health                  -> {ok, coverage_ok}

A background reaper calls registry.tick() so stably-dead nodes free their slots and
under-replication is surfaced (logged / alertable).

Auth: registration is admitted by allowlist + model fingerprint + capacity (in the
Registry). For an open/permissionless network it should ALSO require the node to
sign the request with the ed25519 key behind its node_id — pass `verify_sig=
make_ed25519_verifier()`. Until that's enforced, run permissioned on a private net.
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from engine.log import make_logger
from engine.topology import Node

log = make_logger("control")


def make_ed25519_verifier():
    """A verify_sig(body) that proves the registrant holds the private key behind its
    node_id (= ed25519 public key hex). body must carry `ts` and `sig` (hex); the
    signed message is canonical-JSON of the body minus `sig`."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    def verify(body: dict) -> bool:
        try:
            sig = bytes.fromhex(body["sig"])
            pub = Ed25519PublicKey.from_public_bytes(bytes.fromhex(body["node_id"]))
            msg = json.dumps({k: v for k, v in body.items() if k != "sig"},
                             sort_keys=True, separators=(",", ":")).encode()
            pub.verify(sig, msg)
            return True
        except Exception:
            return False

    return verify


def _handler(registry, now_fn, verify_sig):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read(self):
            n = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(n) or b"{}")

        def do_POST(self):
            try:
                body = self._read()
            except Exception as e:
                return self._send(400, {"error": f"bad body: {e}"})
            try:
                if self.path == "/register":
                    if verify_sig is not None and not verify_sig(body):
                        return self._send(401, {"error": "signature verification failed"})
                    node = Node(
                        node_id=str(body["node_id"]),
                        endpoint=tuple(body["endpoint"]),
                        capacity_layers=int(body["capacity_layers"]),
                        model_fp=str(body.get("model_fp", "")),
                        reachability=str(body.get("reachability", "public")),
                        region=(str(body["region"]) if body.get("region") else None),
                        payout_wallet=str(body.get("payout_wallet", "")),
                    )
                    # loaded_layers (optional): a re-registering node asks for its already-
                    # loaded slot back so it doesn't get a different range and serve stale layers.
                    resp = registry.register(node, now_fn(), prefer_range=body.get("loaded_layers"))
                    log("INFO", "node registered", node=node.node_id[:12],
                        layers=f'{resp["assignment"]["start"]}:{resp["assignment"]["end"]}')
                    return self._send(200, resp)
                if self.path == "/ready":
                    registry.mark_ready(str(body["node_id"]))
                    return self._send(200, {"ok": True})
                if self.path == "/heartbeat":
                    # 'registered' lets a node detect a coordinator restart (it's no longer in
                    # the topology) and re-register without being manually restarted.
                    known = registry.heartbeat(str(body["node_id"]), now_fn())
                    return self._send(200, {"ok": True, "registered": bool(known)})
                if self.path == "/drain":
                    registry.drain(str(body["node_id"]))
                    return self._send(200, {"ok": True})
                return self._send(404, {"error": "not found"})
            except PermissionError as e:
                return self._send(403, {"error": str(e)})
            except ValueError as e:
                return self._send(409, {"error": str(e)})
            except KeyError as e:
                return self._send(400, {"error": f"missing field {e}"})

        def do_GET(self):
            if self.path == "/topology":
                return self._send(200, registry.snapshot())
            if self.path == "/health":
                snap = registry.snapshot()
                return self._send(200, {"ok": True, "coverage_ok": snap["coverage_ok"]})
            return self._send(404, {"error": "not found"})

    return H


def make_server(registry, host="0.0.0.0", port=18932, reap_interval=10.0,
                now_fn=time.time, verify_sig=None,
                rtt_probe_interval: float = 30.0) -> ThreadingHTTPServer:
    """Build the control server + start its background reaper. The caller runs
    srv.serve_forever() (typically in a daemon thread alongside the inference API)."""
    def reaper():
        while True:
            time.sleep(reap_interval)
            try:
                st = registry.tick(now_fn())
                if st["reaped"]:
                    log("WARN", "reaped dead nodes", nodes=st["reaped"])
                if st.get("purged"):
                    log("INFO", "purged retired nodes", nodes=st["purged"])
                if not st["coverage_ok"]:
                    log("WARN", "COVERAGE GAP", needs=st["needs_holders"])
                elif st["needs_holders"]:
                    log("INFO", "under-replicated", needs=st["needs_holders"])
            except Exception as e:  # the reaper must never die
                log("WARN", "reaper error", error=str(e))

    threading.Thread(target=reaper, daemon=True).start()

    # Topology-aware routing: when proximity routing is on, run the active RTT prober so
    # holder ordering uses measured coordinator→node latency (not just region estimates).
    if getattr(getattr(registry, "topo", None), "route_by_latency", False):
        from engine.rtt_probe import start_rtt_prober
        start_rtt_prober(registry, interval=rtt_probe_interval)

    srv = ThreadingHTTPServer((host, port), _handler(registry, now_fn, verify_sig))
    log("INFO", "control channel listening", port=port)
    return srv
