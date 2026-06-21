"""
test_control_channel.py — the node-facing control channel over real HTTP:
register (admission, assignment, per-node key), reject (allowlist / model mismatch),
ready + heartbeat, the /topology view, and ed25519 signature verification.

No GPUs; needs `cryptography` (already a wire dependency):  python3 -m tests.test_control_channel
"""

import json
import os
import socket
import sys
import threading
import urllib.error
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from engine.topology import Topology  # noqa: E402
from engine.registry import Registry, derive_node_key  # noqa: E402
from engine.control_server import make_server, make_ed25519_verifier  # noqa: E402

FP = "qwen2.5-32b-awq@rev1"


def _post(base, path, obj):
    data = json.dumps(obj).encode()
    req = urllib.request.Request(base + path, data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _get(base, path):
    with urllib.request.urlopen(base + path, timeout=5) as r:
        return r.status, json.loads(r.read())


def main():
    topo = Topology(num_layers=64, coordinator_end=32, num_stages=2,
                    model_fp=FP, replication=1, dead_after_s=1e18)
    reg = Registry(topo=topo, master_secret=b"master", coordinator_endpoint=["c", 18931],
                   allowlist={"alice", "bob", "carol"})

    sk = socket.socket(); sk.bind(("127.0.0.1", 0)); port = sk.getsockname()[1]; sk.close()
    srv = make_server(reg, host="127.0.0.1", port=port, reap_interval=1e9)  # no reaping mid-test
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    try:
        # register two allowlisted nodes → slot0, slot1; check assignment + per-node key
        code, r = _post(base, "/register", {"node_id": "alice", "endpoint": ["10.0.0.1", 9000],
                                            "capacity_layers": 16, "model_fp": FP, "payout_wallet": "WA"})
        assert code == 200 and r["assignment"] == {"start": 32, "end": 48}, r
        assert r["session_key"] == derive_node_key(b"master", "alice").hex(), "issued the derived key"
        code, r = _post(base, "/register", {"node_id": "bob", "endpoint": ["10.0.0.2", 9000],
                                            "capacity_layers": 16, "model_fp": FP, "payout_wallet": "WB"})
        assert code == 200 and r["assignment"] == {"start": 48, "end": 64}, r

        # rejections
        code, _ = _post(base, "/register", {"node_id": "eve", "endpoint": ["x", 1],
                                            "capacity_layers": 16, "model_fp": FP})
        assert code == 403, "non-allowlisted → 403"
        code, _ = _post(base, "/register", {"node_id": "carol", "endpoint": ["x", 1],
                                            "capacity_layers": 16, "model_fp": "WRONG-MODEL"})
        assert code == 409, "model mismatch → 409"

        # ready + heartbeat, then the live topology view
        for nid in ("alice", "bob"):
            assert _post(base, "/ready", {"node_id": nid})[0] == 200
            assert _post(base, "/heartbeat", {"node_id": nid})[0] == 200
        code, snap = _get(base, "/topology")
        assert code == 200 and snap["coverage_ok"] is True, snap
        states = {h["node_id"]: h["state"] for s in snap["slots"] for h in s["holders"]}
        assert states == {"alice": "ready", "bob": "ready"}, states

        # ── ed25519 signature verification (proof of key possession) ────────────
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        priv = Ed25519PrivateKey.generate()
        pub_hex = priv.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw).hex()
        body = {"node_id": pub_hex, "endpoint": ["h", 1], "model_fp": FP, "ts": 123}
        msg = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        body["sig"] = priv.sign(msg).hex()
        verify = make_ed25519_verifier()
        assert verify(body) is True, "valid signature accepted"
        tampered = dict(body); tampered["model_fp"] = "TAMPERED"
        assert verify(tampered) is False, "tampered body rejected"

        print("CONTROL CHANNEL TESTS PASSED — register/assign/key, reject, ready, topology, ed25519")
    finally:
        srv.shutdown()


if __name__ == "__main__":
    main()
