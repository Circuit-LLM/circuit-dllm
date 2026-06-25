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


def make_ed25519_signer(private_key_hex: str, node_id_hex: str, now_fn=time.time):
    """The orchestrator side of make_ed25519_verifier: stamp `node_id`+`ts`, sign canonical-JSON of
    the body (minus `sig`) with the ed25519 private key, attach `sig`. node_id_hex MUST be the
    public key behind private_key_hex (node_id == ed25519 pubkey)."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    priv = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(private_key_hex))

    def sign(body: dict) -> dict:
        body = {**body, "node_id": node_id_hex, "ts": now_fn()}
        msg = json.dumps({k: v for k, v in body.items() if k != "sig"},
                         sort_keys=True, separators=(",", ":")).encode()
        body["sig"] = priv.sign(msg).hex()
        return body

    return sign


def _node_json(registry, n, with_key=False):
    """Serialize a Node to route metadata for the floating-coordinator RPCs. `layers` is the node's
    assigned slice [start, end), or None for a head-only orchestrator. The data-wire `wire_key` is
    included ONLY when with_key (an AUTHENTICATED orchestrator's route) — it's the credential to
    talk to that slice; never hand it to an unauthenticated caller."""
    sl = None
    if n.slot is not None and 0 <= n.slot < len(registry.topo.slots):
        s = registry.topo.slots[n.slot]
        sl = [s.start, s.end]
    out = {"node_id": n.node_id, "endpoint": list(n.endpoint), "layers": sl,
           "reachability": n.reachability, "trust": n.trust, "orchestrator": n.orchestrator}
    if with_key and n.wire_key is not None:
        out["wire_key"] = n.wire_key.hex()
    return out


def _handler(registry, now_fn, verify_sig, entry_require_sig=False):
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
                        orchestrator=bool(body.get("orchestrator", False)),
                    )
                    # loaded_layers (optional): a re-registering node asks for its already-
                    # loaded slot back so it doesn't get a different range and serve stale layers.
                    resp = registry.register(node, now_fn(), prefer_range=body.get("loaded_layers"))
                    a = resp.get("assignment")
                    log("INFO", "node registered", node=node.node_id[:12],
                        layers=(f'{a["start"]}:{a["end"]}' if a else "orchestrator(head-only)"))
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
                # ── floating-coordinator route/entry RPCs (docs/FLOATING_COORDINATOR.md) ──────────
                # AUTHENTICATED: a remote orchestrator proves identity with an ed25519-signed body.
                # Without verify_sig configured these are unavailable (403), so deploying them exposes
                # nothing until orchestrator auth is turned on. /route/acquire returns the data-plane
                # wire keys ONLY to a verified caller.
                # ── entry RPCs: orchestrator SELECTION (acquire_entry/release_entry) ──────────────
                # These return only PUBLIC orchestrator endpoints (no wire keys), so they're OPEN by
                # default — the stateless gateway calls them once per request with no signing. Set
                # CIRCUIT_ENTRY_REQUIRE_SIG=1 to gate them like the route RPCs (e.g. if the control
                # plane is internet-exposed and entry-load-skew abuse is a concern).
                if self.path in ("/entry/acquire", "/entry/release"):
                    if entry_require_sig and (verify_sig is None or not verify_sig(body)):
                        return self._send(401, {"error": "entry RPCs require a valid signature"})
                    session = str(body["session"])
                    if self.path == "/entry/acquire":
                        o = registry.acquire_entry(session)
                        return self._send(200, {"orchestrator": (_node_json(registry, o) if o else None)})
                    registry.release_entry(session)            # /entry/release
                    return self._send(200, {"ok": True})
                # ── route RPCs: hand out data-wire KEYS / failover → ALWAYS authenticated ─────────
                if self.path in ("/route/acquire", "/route/release", "/route/suspect"):
                    if verify_sig is None:
                        return self._send(403, {"error": "route RPCs require CIRCUIT_MESH_VERIFY_SIG=1"})
                    if not verify_sig(body):
                        return self._send(401, {"error": "signature verification failed"})
                    if self.path == "/route/suspect":      # failover: a remote orchestrator reports a dead holder
                        registry.mark_suspect(str(body["suspect"]))
                        return self._send(200, {"ok": True})
                    session = str(body["session"])
                    if self.path == "/route/acquire":
                        try:
                            route = registry.acquire_route(session)
                        except RuntimeError as e:              # coverage gap
                            return self._send(503, {"error": str(e)})
                        return self._send(200, {"route": [_node_json(registry, n, with_key=True) for n in route]})
                    registry.release_route(session)            # /route/release
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
            if self.path == "/payouts/eligible":   # live nodes + payout wallets (the payout executor polls this)
                return self._send(200, {"nodes": registry.payout_eligible()})
            return self._send(404, {"error": "not found"})

    return H


def make_server(registry, host="0.0.0.0", port=18932, reap_interval=10.0,
                now_fn=time.time, verify_sig=None,
                rtt_probe_interval: float = 30.0,
                entry_require_sig: bool = False) -> ThreadingHTTPServer:
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

    srv = ThreadingHTTPServer((host, port), _handler(registry, now_fn, verify_sig, entry_require_sig))
    log("INFO", "control channel listening", port=port)
    return srv
