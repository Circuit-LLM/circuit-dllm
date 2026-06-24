"""
test_relay.py — NAT relay rendezvous (docs/RELAY.md), end-to-end over loopback.

A simulated NAT node (outbound control + on-demand data conns, real ed25519 auth) and a
coordinator dialing through the relay must exchange bytes BOTH directions — proving the relay
pairs and pipes correctly. Also checks: bad coordinator token is rejected; dialing an offline
node fails fast. No torch / no GPU.

Run:  python3 -m tests.test_relay
"""

import os
import socket
import sys
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from cryptography.hazmat.primitives import serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # noqa: E402

from engine.relay import RelayServer, read_preamble, write_preamble  # noqa: E402

TOKEN = "coord-secret"


def _free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close()
    return p


def _node_identity():
    sk = Ed25519PrivateKey.generate()
    pub = sk.public_key().public_bytes(serialization.Encoding.Raw,
                                        serialization.PublicFormat.Raw)
    return sk, pub.hex()


def _connect(port):
    c = socket.create_connection(("127.0.0.1", port), timeout=5)
    c.settimeout(5)
    return c


def _run_node(port, sk, node_id, ready_evt):
    """A NAT node: outbound control conn, auth, then on each OPEN open a data conn that echoes."""
    ctrl = _connect(port)
    write_preamble(ctrl, {"role": "control", "node_id": node_id})
    nonce = bytes.fromhex(read_preamble(ctrl)["nonce"])
    write_preamble(ctrl, {"sig": sk.sign(nonce).hex()})
    assert read_preamble(ctrl).get("ok") is True, "control auth failed"
    ready_evt.set()
    while True:
        try:
            msg = read_preamble(ctrl)
        except Exception:
            return
        if msg.get("op") == "open":
            tag = msg["tag"]
            data = _connect(port)
            write_preamble(data, {"role": "data", "node_id": node_id, "tag": tag})
            threading.Thread(target=_echo, args=(data,), daemon=True).start()


def _echo(sock):
    try:
        while True:
            b = sock.recv(4096)
            if not b:
                break
            sock.sendall(b)            # echo back: proves duplex piping
    except OSError:
        pass


def main():
    port = _free_port()
    relay = RelayServer("127.0.0.1", port, coord_token=TOKEN)
    threading.Thread(target=relay.serve_forever, daemon=True).start()
    time.sleep(0.2)

    sk, node_id = _node_identity()
    ready = threading.Event()
    threading.Thread(target=_run_node, args=(port, sk, node_id, ready), daemon=True).start()
    assert ready.wait(5), "node never came online"
    time.sleep(0.1)
    assert node_id in relay.nodes_online(), "node not registered on control channel"

    # coordinator dials through the relay and round-trips bytes through the node
    coord = _connect(port)
    write_preamble(coord, {"role": "dial", "node_id": node_id, "token": TOKEN})
    assert read_preamble(coord).get("ok") is True, "dial not accepted"
    payload = b"ACTIVATION-ciphertext-" + os.urandom(32)
    coord.sendall(payload)
    got = b""
    while len(got) < len(payload):
        chunk = coord.recv(4096)
        if not chunk:
            break
        got += chunk
    assert got == payload, f"round-trip mismatch: {len(got)}/{len(payload)} bytes"

    # bad coordinator token is rejected
    bad = _connect(port)
    write_preamble(bad, {"role": "dial", "node_id": node_id, "token": "wrong"})
    assert read_preamble(bad).get("ok") is False, "bad token must be rejected"

    # dialing an unknown/offline node fails fast (no pairing hang)
    off = _connect(port)
    write_preamble(off, {"role": "dial", "node_id": "deadbeef", "token": TOKEN})
    assert read_preamble(off).get("ok") is False, "offline node dial must fail"

    relay.stop()
    print("RELAY TESTS PASSED — ed25519 control auth, dial→open→data pairing, duplex piping, "
          "token + offline rejection")


if __name__ == "__main__":
    main()
