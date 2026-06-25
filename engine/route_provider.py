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


class LocalRouteProvider:
    """In-process registry — today's behavior, byte-identical. The orchestrator IS the coordinator,
    so it already holds the registry and the wire keys; this just normalizes to RouteHops."""

    def __init__(self, registry):
        self._reg = registry

    def acquire(self, session) -> List[RouteHop]:
        hops: List[RouteHop] = []
        for n in self._reg.acquire_route(session):          # raises on a coverage gap, as today
            sl = None
            if n.slot is not None and 0 <= n.slot < len(self._reg.topo.slots):
                s = self._reg.topo.slots[n.slot]
                sl = (s.start, s.end)
            hops.append(RouteHop(node_id=n.node_id, host=n.endpoint[0], port=n.endpoint[1],
                                 layers=sl, reachability=n.reachability, wire_key=n.wire_key))
        return hops

    def release(self, session) -> None:
        self._reg.release_route(session)


class RemoteRouteProvider:
    """Head-only orchestrator → standalone control plane over the authenticated control channel.
    `sign` is a control_server.make_ed25519_signer(...) that stamps node_id+ts+sig on each body."""

    def __init__(self, control_url: str, sign, timeout: float = 8.0):
        self._url = control_url.rstrip("/")
        self._sign = sign
        self._timeout = timeout

    def _post(self, path: str, body: dict) -> dict:
        data = json.dumps(self._sign(dict(body))).encode()
        req = urllib.request.Request(self._url + path, data=data,
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=self._timeout) as r:
            return json.loads(r.read() or b"{}")

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
