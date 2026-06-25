"""RouteProvider — where a session's slice route comes from.

The head/orchestrator shouldn't care whether the routing registry is in-process (today) or a remote
control plane (floating coordinator). Both providers return the SAME `RouteHop` list, so the decode
loop dials + encrypts identically. `LocalRouteProvider` is byte-identical to the current path;
`RemoteRouteProvider` is what a head-only node uses against a standalone control plane over the
authenticated control channel. See docs/FLOATING_COORDINATOR.md.

Pure stdlib (no torch / no third-party HTTP) so it unit-tests on a CPU box.
"""
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class RouteHop:
    """One slice in a session's route: where it is + the data-wire key to talk to it."""
    node_id: str
    host: Optional[str]
    port: Optional[int]
    layers: Optional[Tuple[int, int]]   # (start, end), or None
    reachability: str = "public"
    wire_key: Optional[bytes] = None    # None until an authenticated control plane provides it

    @property
    def endpoint(self) -> Tuple[Optional[str], Optional[int]]:
        """(host, port) — so a RouteHop is a drop-in for a topology Node in the relay path: the
        in-process Coordinator consumes Node.endpoint; a head-only orchestrator consumes
        RouteHop.endpoint identically. `node_id`/`reachability`/`wire_key` already line up."""
        return (self.host, self.port)


class LocalRouteProvider:
    """In-process registry — today's behavior, BYTE-IDENTICAL. The orchestrator IS the coordinator,
    so it already holds the registry + wire keys. This is a pure passthrough: it returns the
    registry's Node objects unchanged, so the Coordinator's existing relay/failover/reset code (which
    consumes Nodes) is untouched — there is no behavioral change at replication=1 or otherwise.

    NOTE on the contract: LocalRouteProvider.acquire returns Node objects (in-process Coordinator);
    RemoteRouteProvider.acquire returns RouteHops (the future head-only Orchestrator). Both are
    "a dialable route"; unifying the Coordinator onto RouteHops is a follow-up done alongside the
    remote-orchestrator build (they share the RouteHop consumer)."""

    def __init__(self, registry):
        self._reg = registry

    def acquire(self, session):
        return self._reg.acquire_route(session)   # Node objects, exactly as the Coordinator expects

    def release(self, session) -> None:
        self._reg.release_route(session)

    def mark_suspect(self, node_id) -> None:
        """Failover: tell the (in-process) registry a holder is dead so the next acquire routes
        around it. Identical to today's direct registry.mark_suspect call it replaces."""
        self._reg.mark_suspect(node_id)

    def to_hops(self, nodes) -> List[RouteHop]:
        """Normalize Node objects → RouteHops (for callers that want the wire form; not used by the
        in-process relay). Kept here so the Node→RouteHop mapping lives in one place."""
        hops: List[RouteHop] = []
        for n in nodes:
            sl = None
            if n.slot is not None and 0 <= n.slot < len(self._reg.topo.slots):
                s = self._reg.topo.slots[n.slot]
                sl = (s.start, s.end)
            hops.append(RouteHop(node_id=n.node_id, host=n.endpoint[0], port=n.endpoint[1],
                                 layers=sl, reachability=n.reachability, wire_key=n.wire_key))
        return hops


class ControlEndpoints:
    """A failover-aware client for one-OR-MORE standalone control planes.

    The control plane is no longer a single process (the SPOF in docs/CUTOVER_FLOATING.md):
    an active + one or more warm standbys run on different pods (docs/CONTROL_PLANE_HA.md).
    Clients (orchestrators, holders, the gateway) hold the LIST and use:
      * ``post``     — FAILOVER: try the sticky last-good url, then the rest in order; the
                       first that answers becomes preferred. For per-request RPCs (route /
                       heartbeat) where any one live control plane is enough.
      * ``post_all`` — best-effort fan-out to EVERY url so register/ready keep all control
                       planes warm; a standby is then ready to take over with no cold start.

    Pure stdlib so it unit-tests on a CPU box. ``sign``, if given, is applied to a FRESH copy
    of the body per attempt, so each POST carries a fresh ts+sig and can't drift outside the
    control plane's replay window during a slow failover."""

    def __init__(self, urls, sign=None, timeout: float = 8.0):
        if isinstance(urls, str):
            urls = [u.strip() for u in urls.split(",")]
        self._urls = [u.rstrip("/") for u in urls if u and u.strip()]
        if not self._urls:
            raise ValueError("ControlEndpoints needs at least one url")
        self._sign = sign
        self._timeout = timeout
        self._preferred = 0                       # index of the last url that answered (sticky)

    @property
    def urls(self) -> List[str]:
        return list(self._urls)

    @property
    def preferred(self) -> str:
        return self._urls[self._preferred]

    def _post_one(self, url: str, path: str, body: dict, timeout: float) -> Tuple[int, dict]:
        data = json.dumps(self._sign(dict(body)) if self._sign else dict(body)).encode()
        req = urllib.request.Request(url + path, data=data,
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return getattr(r, "status", 200), json.loads(r.read() or b"{}")

    def post(self, path: str, body: dict, timeout: Optional[float] = None) -> Tuple[int, dict]:
        """Failover POST. Tries the sticky-preferred url first, then the others; the first to
        answer becomes preferred. Raises the last error only if EVERY control plane fails."""
        t = self._timeout if timeout is None else timeout
        order = [self._preferred] + [i for i in range(len(self._urls)) if i != self._preferred]
        last: Optional[Exception] = None
        for i in order:
            try:
                st, resp = self._post_one(self._urls[i], path, body, t)
                self._preferred = i                # stick to whoever answered
                return st, resp
            except Exception as e:                 # noqa: BLE001 — fall through to the next control plane
                last = e
        raise last if last else RuntimeError("no control endpoints")

    def post_all(self, path: str, body: dict, timeout: Optional[float] = None):
        """Best-effort fan-out to EVERY control plane (keeps standbys warm). Never raises.
        Returns [(url, status|None, parsed|None, err|None)] in url order."""
        t = self._timeout if timeout is None else timeout
        out = []
        for u in self._urls:
            try:
                st, resp = self._post_one(u, path, body, t)
                out.append((u, st, resp, None))
            except Exception as e:                 # noqa: BLE001
                out.append((u, None, None, e))
        return out


class RemoteRouteProvider:
    """Head-only orchestrator → standalone control plane over the authenticated control channel.
    `sign` is a control_server.make_ed25519_signer(...) that stamps node_id+ts+sig on each body.
    `control` is a ControlEndpoints, or a url / comma-list / [urls] (+ sign) for back-compat — so
    a multi-control-plane orchestrator fails routing over to a standby transparently."""

    def __init__(self, control, sign=None, timeout: float = 8.0):
        self._eps = control if isinstance(control, ControlEndpoints) \
            else ControlEndpoints(control, sign=sign, timeout=timeout)

    def _post(self, path: str, body: dict) -> dict:
        _st, resp = self._eps.post(path, body)
        return resp

    def acquire(self, session) -> List[RouteHop]:
        resp = self._post("/route/acquire", {"session": str(session)})
        hops: List[RouteHop] = []
        for h in resp.get("route", []):
            ep = h.get("endpoint") or [None, None]
            ly = tuple(h["layers"]) if h.get("layers") else None
            wk = bytes.fromhex(h["wire_key"]) if h.get("wire_key") else None
            hops.append(RouteHop(node_id=h["node_id"], host=ep[0], port=ep[1], layers=ly,
                                 reachability=h.get("reachability", "public"), wire_key=wk))
        return hops

    def release(self, session) -> None:
        try:
            self._post("/route/release", {"session": str(session)})
        except Exception:
            pass   # best-effort; the control plane also reaps a session on its load-timeout

    def mark_suspect(self, node_id) -> None:
        """Report a dead/misbehaving holder so the control plane's next acquire_route routes
        around it. The target goes in `suspect`, NOT `node_id` — the signer overwrites `node_id`
        with the CALLER's identity. Best-effort: the control plane also reaps on heartbeat timeout."""
        try:
            self._post("/route/suspect", {"suspect": str(node_id)})
        except Exception:
            pass
