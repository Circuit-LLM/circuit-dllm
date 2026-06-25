"""RouteProvider Phase 2 — Local (byte-identical) + Remote (authenticated round-trip over HTTP).

Pure logic (no torch): stands up the real control_server in a thread and drives it with the real
RemoteRouteProvider + ed25519 signing, proving the auth gate and the wire-key hand-off end to end.
"""
import json
import threading
import urllib.request
import urllib.error

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import (
    Encoding, PrivateFormat, PublicFormat, NoEncryption)

from engine.topology import Topology, Node
from engine.registry import Registry
from engine.control_server import make_server, make_ed25519_verifier, make_ed25519_signer
from engine.route_provider import LocalRouteProvider, RemoteRouteProvider


def _covered_registry():
    """A registry whose 3 stage slots are all covered + READY (so acquire_route succeeds)."""
    topo = Topology(num_layers=80, coordinator_end=20, num_stages=3, model_fp="t", replication=1)
    reg = Registry(topo=topo, master_secret=b"master" * 8, coordinator_endpoint=("coord", 1))
    for i in range(3):
        nid = f"stage{i}"
        reg.register(Node(node_id=nid, endpoint=(f"10.0.0.{i}", 5000 + i), capacity_layers=20,
                          model_fp="t"))
        reg.mark_ready(nid)
    assert reg.snapshot()["coverage_ok"], "test setup: slots not covered"
    return reg


def _ed25519_pair():
    p = Ed25519PrivateKey.generate()
    priv = p.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption()).hex()
    pub = p.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
    return priv, pub


def _serve(reg, verify_sig):
    srv = make_server(reg, host="127.0.0.1", port=0, reap_interval=3600, verify_sig=verify_sig)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def test_local_provider_byte_identical():
    reg = _covered_registry()
    hops = LocalRouteProvider(reg).acquire("s1")
    assert len(hops) == 3
    assert all(h.wire_key is not None for h in hops)             # local already holds the keys
    assert all(h.layers is not None for h in hops)
    LocalRouteProvider(reg).release("s1")
    assert reg.load_snapshot() == {}                            # released


def test_remote_round_trip_with_keys():
    reg = _covered_registry()
    srv, url = _serve(reg, make_ed25519_verifier())
    try:
        priv, pub = _ed25519_pair()
        rp = RemoteRouteProvider(url, make_ed25519_signer(priv, pub))
        hops = rp.acquire("sX")
        assert len(hops) == 3
        assert all(h.wire_key is not None and len(h.wire_key) == 32 for h in hops)  # keys handed to authed caller
        assert {h.node_id for h in hops} == {"stage0", "stage1", "stage2"}
        # the remote-acquired keys match what the registry derived locally
        local = {h.node_id: h.wire_key for h in LocalRouteProvider(reg).acquire("local")}
        assert all(h.wire_key == local[h.node_id] for h in hops)
        rp.release("sX")
    finally:
        srv.shutdown()


def test_auth_gate_403_without_verify_sig():
    reg = _covered_registry()
    srv, url = _serve(reg, None)                                 # verify_sig NOT configured
    try:
        body = json.dumps({"session": "s"}).encode()
        try:
            urllib.request.urlopen(urllib.request.Request(url + "/route/acquire", data=body,
                                   headers={"Content-Type": "application/json"}, method="POST"))
            assert False, "expected 403"
        except urllib.error.HTTPError as e:
            assert e.code == 403
    finally:
        srv.shutdown()


def test_auth_gate_401_on_bad_signature():
    reg = _covered_registry()
    srv, url = _serve(reg, make_ed25519_verifier())
    try:
        # a body claiming a node_id but with a garbage sig
        bad = json.dumps({"session": "s", "node_id": _ed25519_pair()[1], "ts": 1, "sig": "00" * 64}).encode()
        try:
            urllib.request.urlopen(urllib.request.Request(url + "/route/acquire", data=bad,
                                   headers={"Content-Type": "application/json"}, method="POST"))
            assert False, "expected 401"
        except urllib.error.HTTPError as e:
            assert e.code == 401
    finally:
        srv.shutdown()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"  ok  {fn.__name__}")
    print(f"\nALL {len(fns)} route-provider Phase-2 tests passed")
