"""
relay.py — NAT relay / rendezvous for home-desktop GPU nodes (docs/RELAY.md).

The coordinator DIALS into stages, so a node behind NAT (a home router) is unreachable. This
relay fixes that without port-forwarding: a node opens an OUTBOUND control connection to the
relay; when the coordinator wants that node it connects to the RELAY, which signals the node to
open an outbound DATA connection, then **pipes raw bytes** between the two (ngrok/frp model).

The relay is untrusted plumbing: the engine data wire is ChaCha20-encrypted end-to-end under the
node's per-node key, so the relay only ever copies CIPHERTEXT — it can't read or forge activations.
Node identity on the control channel is proven by an ed25519 signature over a relay nonce (node_id
== the public key, same identity it registers with the coordinator), so a node can't squat another's
id. Coordinators present a shared `coord_token`.

Protocol — each connection opens with one length-prefixed JSON preamble, then becomes a raw byte
pipe. Roles:
  control : node→relay, persistent. {role:"control", node_id}; relay replies {nonce}; node sends
            {sig}; relay verifies. Relay pushes {op:"open", tag} here when a coordinator wants it.
  dial    : coordinator→relay. {role:"dial", node_id, token}; relay mints a tag, signals the node,
            and on the node's matching data conn pipes this socket to it.
  data    : node→relay, on demand. {role:"data", node_id, tag}; relay pairs it with the waiting
            dial socket for (node_id, tag) and pipes.

Pure stdlib + cryptography (already an engine dep). Threaded; unit-testable over loopback.
"""

from __future__ import annotations

import json
import os
import secrets
import socket
import struct
import threading
from typing import Dict, Optional, Tuple

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

_PREAMBLE_MAX = 64 * 1024
_DIAL_WAIT_S = 30.0       # how long a dialing coordinator waits for the node's data conn
_PIPE_BUF = 1 << 16


# ── framing: one length-prefixed JSON object, then raw bytes ──────────────────
def _recvall(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed")
        buf += chunk
    return bytes(buf)


def read_preamble(sock: socket.socket) -> dict:
    (length,) = struct.unpack(">I", _recvall(sock, 4))
    if length > _PREAMBLE_MAX:
        raise ValueError(f"preamble too large: {length}")
    return json.loads(_recvall(sock, length).decode("utf-8"))


def write_preamble(sock: socket.socket, obj: dict) -> None:
    body = json.dumps(obj).encode("utf-8")
    sock.sendall(struct.pack(">I", len(body)) + body)


def _verify_node_sig(node_id: str, nonce: bytes, sig_hex: str) -> bool:
    """node_id is the ed25519 public key (hex); a node proves it holds the matching private key
    by signing the relay's nonce. Same identity scheme as coordinator /register."""
    try:
        pub = Ed25519PublicKey.from_public_bytes(bytes.fromhex(node_id))
        pub.verify(bytes.fromhex(sig_hex), nonce)
        return True
    except (InvalidSignature, ValueError):
        return False


def _pipe(a: socket.socket, b: socket.socket) -> None:
    """Copy a→b until EOF/error, then half-close b. One direction; spawn two for full duplex."""
    try:
        while True:
            data = a.recv(_PIPE_BUF)
            if not data:
                break
            b.sendall(data)
    except OSError:
        pass
    finally:
        for s in (a, b):
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


class RelayServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 18940,
                 coord_token: str = "", verify_sig: bool = True, log=None):
        self.host, self.port = host, port
        self.coord_token = coord_token
        self.verify_sig = verify_sig
        self.log = log or (lambda *a, **k: None)
        self._controls: Dict[str, Tuple[socket.socket, threading.Lock]] = {}  # node_id -> (sock, write-lock)
        self._pending: Dict[Tuple[str, str], Tuple[socket.socket, threading.Event]] = {}
        self._lock = threading.Lock()
        self._srv: Optional[socket.socket] = None
        self._stop = threading.Event()

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def serve_forever(self) -> None:
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind((self.host, self.port))
        self._srv.listen(128)
        self.log("INFO", "relay listening", port=self.port)
        while not self._stop.is_set():
            try:
                conn, _ = self._srv.accept()
            except OSError:
                break
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def stop(self) -> None:
        self._stop.set()
        if self._srv is not None:
            try:
                self._srv.close()
            except OSError:
                pass

    def nodes_online(self):
        with self._lock:
            return sorted(self._controls)

    # ── dispatch ──────────────────────────────────────────────────────────────
    def _handle(self, conn: socket.socket) -> None:
        handed_off = False
        try:
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            pre = read_preamble(conn)
            role = pre.get("role")
            if role == "control":
                self._handle_control(conn, pre.get("node_id", ""))
            elif role == "dial":
                handed_off = self._handle_dial(conn, pre.get("node_id", ""), pre.get("token", ""))
            elif role == "data":
                handed_off = self._handle_data(conn, pre.get("node_id", ""), pre.get("tag", ""))
            else:
                write_preamble(conn, {"ok": False, "err": "unknown role"})
        except (OSError, ValueError, ConnectionError) as e:
            self.log("WARN", "relay conn error", err=str(e))
        finally:
            if not handed_off:
                try:
                    conn.close()
                except OSError:
                    pass

    # ── control: a node registers and stays connected for OPEN signals ─────────
    def _handle_control(self, conn: socket.socket, node_id: str) -> None:
        if not node_id:
            write_preamble(conn, {"ok": False, "err": "no node_id"}); return
        nonce = secrets.token_bytes(32)
        write_preamble(conn, {"nonce": nonce.hex()})
        resp = read_preamble(conn)
        if self.verify_sig and not _verify_node_sig(node_id, nonce, resp.get("sig", "")):
            write_preamble(conn, {"ok": False, "err": "bad signature"}); return
        write_preamble(conn, {"ok": True})
        with self._lock:
            old = self._controls.pop(node_id, None)
            self._controls[node_id] = (conn, threading.Lock())
        if old is not None:                       # a reconnecting node replaces its prior control conn
            try:
                old[0].close()
            except OSError:
                pass
        self.log("INFO", "relay node online", node=node_id[:8])
        try:
            while not self._stop.is_set():        # hold open; reads detect close (+ optional PONGs)
                if not conn.recv(_PIPE_BUF):
                    break
        except OSError:
            pass
        finally:
            with self._lock:
                cur = self._controls.get(node_id)
                if cur and cur[0] is conn:
                    del self._controls[node_id]
            self.log("INFO", "relay node offline", node=node_id[:8])

    def _signal_open(self, node_id: str, tag: str) -> bool:
        with self._lock:
            entry = self._controls.get(node_id)
        if entry is None:
            return False
        sock, wlock = entry
        try:
            with wlock:
                write_preamble(sock, {"op": "open", "tag": tag})
            return True
        except OSError:
            return False

    # ── dial: a coordinator asks for node_id; wait for its data conn, then pipe ─
    def _handle_dial(self, conn: socket.socket, node_id: str, token: str) -> bool:
        if self.coord_token and token != self.coord_token:
            write_preamble(conn, {"ok": False, "err": "bad token"}); return False
        tag = secrets.token_hex(16)
        ev = threading.Event()
        with self._lock:
            self._pending[(node_id, tag)] = (conn, ev)
        if not self._signal_open(node_id, tag):
            with self._lock:
                self._pending.pop((node_id, tag), None)
            write_preamble(conn, {"ok": False, "err": "node offline"}); return False
        write_preamble(conn, {"ok": True})        # tell the coordinator to start speaking the wire
        if ev.wait(_DIAL_WAIT_S):
            return True                           # paired — the data handler owns this socket now
        with self._lock:
            self._pending.pop((node_id, tag), None)
        return False                              # timed out → _handle closes conn

    # ── data: the node's on-demand conn; pair with the waiting dial socket ─────
    def _handle_data(self, conn: socket.socket, node_id: str, tag: str) -> bool:
        with self._lock:
            pend = self._pending.pop((node_id, tag), None)
        if pend is None:
            return False                          # no waiter (timed out / bad tag) → close
        coord_sock, ev = pend
        ev.set()                                  # release the dial handler (it must NOT close coord_sock)
        # full-duplex pipe; each direction half-closes on EOF, freeing both sockets.
        threading.Thread(target=_pipe, args=(conn, coord_sock), daemon=True).start()
        threading.Thread(target=_pipe, args=(coord_sock, conn), daemon=True).start()
        return True


def main(argv) -> int:
    import argparse
    from engine.log import make_logger
    ap = argparse.ArgumentParser("circuit-relay")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=int(os.environ.get("CIRCUIT_RELAY_PORT", "18940")))
    ap.add_argument("--coord-token", default=os.environ.get("CIRCUIT_RELAY_TOKEN", ""))
    ap.add_argument("--no-verify", action="store_true", help="skip node sig check (testing only)")
    a = ap.parse_args(argv)
    log = make_logger("relay")
    RelayServer(a.host, a.port, coord_token=a.coord_token,
                verify_sig=not a.no_verify, log=log).serve_forever()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv[1:]))
