"""ControlEndpoints — multi-control-plane failover (docs/CONTROL_PLANE_HA.md).

Stands up TWO real control_server instances (active + warm standby) and drives them with the real
ControlEndpoints / RemoteRouteProvider + ed25519 signing. Proves: failover sticks to whoever answers,
post_all fans out best-effort, a head-only orchestrator routes through a standby when the active is
dead, and an all-dead set raises. Pure logic (no torch) — runs on a CPU box.
"""
import threading

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import (
    Encoding, PrivateFormat, PublicFormat, NoEncryption)

from engine.topology import Topology, Node
from engine.registry import Registry
from engine.control_server import make_server, make_ed25519_verifier, make_ed25519_signer
from engine.route_provider import ControlEndpoints, RemoteRouteProvider


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


def _serve():
    """A standalone control plane (covered + ready) in a thread. Returns (srv, url)."""
    reg = _covered_registry()
    srv = make_server(reg, host="127.0.0.1", port=0, reap_interval=3600,
                      verify_sig=make_ed25519_verifier())
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


DEAD = "http://127.0.0.1:1"          # nothing listens on port 1 → connection refused fast


def test_parses_comma_string_and_normalizes():
    eps = ControlEndpoints("http://a:1/ , http://b:2//", timeout=1.0)
    assert eps.urls == ["http://a:1", "http://b:2"]
    assert eps.preferred == "http://a:1"


def test_empty_urls_rejected():
    for bad in ["", "  ,  ", []]:
        try:
            ControlEndpoints(bad)
            assert False, f"expected ValueError for {bad!r}"
        except ValueError:
            pass


def test_failover_sticks_to_whoever_answers():
    srv1, url1 = _serve()
    srv2, url2 = _serve()
    priv, pub = _ed25519_pair()
    eps = ControlEndpoints([url1, url2], sign=make_ed25519_signer(priv, pub), timeout=2.0)
    try:
        st, resp = eps.post("/route/acquire", {"session": "s1"})
        assert st == 200 and len(resp["route"]) == 3
        assert eps.preferred == url1                       # active answered → preferred

        srv1.shutdown()                                    # active dies
        st, resp = eps.post("/route/acquire", {"session": "s2"})
        assert st == 200 and len(resp["route"]) == 3       # served by the standby
        assert eps.preferred == url2                       # now sticky to the standby
    finally:
        srv2.shutdown()
        try: srv1.shutdown()
        except Exception: pass


def test_post_all_is_best_effort():
    srv1, url1 = _serve()
    srv2, url2 = _serve()
    priv, pub = _ed25519_pair()
    eps = ControlEndpoints([url1, url2], sign=make_ed25519_signer(priv, pub), timeout=2.0)
    try:
        # both up → both answer
        results = eps.post_all("/route/acquire", {"session": "a"})
        assert [r[3] for r in results] == [None, None] and all(r[1] == 200 for r in results)

        srv1.shutdown()                                    # one down → one ok, one err, NO raise
        results = eps.post_all("/route/acquire", {"session": "b"})
        oks = [r for r in results if r[3] is None]
        errs = [r for r in results if r[3] is not None]
        assert len(oks) == 1 and len(errs) == 1 and oks[0][0] == url2
    finally:
        srv2.shutdown()
        try: srv1.shutdown()
        except Exception: pass


def test_remote_route_provider_routes_through_standby():
    srv2, url2 = _serve()
    priv, pub = _ed25519_pair()
    # active is DEAD from the start; the orchestrator must still get a route from the standby.
    rp = RemoteRouteProvider(ControlEndpoints([DEAD, url2], sign=make_ed25519_signer(priv, pub),
                                              timeout=2.0))
    try:
        hops = rp.acquire("s3")
        assert len(hops) == 3 and all(h.wire_key is not None for h in hops)
        rp.release("s3")                                   # best-effort, must not raise
        rp.mark_suspect("stage0")                          # best-effort, must not raise
    finally:
        srv2.shutdown()


def _norm_slots(reg):
    """Slot layout + holders (node_id, state), rtt-independent — for comparing two control planes."""
    return [(s["slot"], tuple(s["layers"]),
             tuple(sorted((h["node_id"], h["state"]) for h in s["holders"])))
            for s in reg.snapshot()["slots"]]


def test_holders_mirror_to_standby_identical_and_routable():
    """The crux of the no-SOL HA design: a holder registers its slot on the active, then mirrors to
    the standby with prefer_range → the standby ends up with an IDENTICAL, independently-routable
    topology. Kill the active and a head-only orchestrator still gets a full route from the standby."""
    def _empty():
        topo = Topology(num_layers=80, coordinator_end=20, num_stages=3, model_fp="t", replication=1)
        reg = Registry(topo=topo, master_secret=b"master" * 8, coordinator_endpoint=("coord", 1))
        srv = make_server(reg, host="127.0.0.1", port=0, reap_interval=3600,
                          verify_sig=make_ed25519_verifier())
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return reg, srv, f"http://127.0.0.1:{srv.server_address[1]}"

    reg1, srv1, url1 = _empty()
    reg2, srv2, url2 = _empty()
    try:
        for i in range(3):
            priv, pub = _ed25519_pair()
            eps_h = ControlEndpoints([url1, url2], sign=make_ed25519_signer(priv, pub), timeout=2.0)
            body = {"endpoint": [f"10.0.0.{i}", 5000 + i], "capacity_layers": 20,
                    "model_fp": "t", "reachability": "public"}
            _st, resp = eps_h.post("/register", dict(body))             # active assigns a slot
            asn = resp["assignment"]
            mirror = dict(body); mirror["loaded_layers"] = [asn["start"], asn["end"]]
            _st2, resp2 = eps_h._post_one(url2, "/register", mirror, 2.0)  # mirror w/ prefer_range
            assert resp2["assignment"] == asn, "standby must honor prefer_range (identical slot)"
            eps_h.post_all("/ready", {"node_id": pub})

        assert reg1.snapshot()["coverage_ok"] and reg2.snapshot()["coverage_ok"]
        assert _norm_slots(reg1) == _norm_slots(reg2), "standby topology must match the active"

        srv1.shutdown()                                                # active dies
        opriv, opub = _ed25519_pair()
        rp = RemoteRouteProvider(ControlEndpoints([url1, url2], sign=make_ed25519_signer(opriv, opub),
                                                  timeout=2.0))
        hops = rp.acquire("s1")                                        # routed by the standby alone
        assert len(hops) == 3 and all(h.wire_key is not None for h in hops)
    finally:
        srv2.shutdown()
        try: srv1.shutdown()
        except Exception: pass


def test_all_dead_raises():
    eps = ControlEndpoints([DEAD, "http://127.0.0.1:2"], timeout=1.0)
    try:
        eps.post("/route/acquire", {"session": "x"})
        assert False, "expected an error when every control plane is dead"
    except Exception:
        pass


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"  ok  {fn.__name__}")
    print(f"\nALL {len(fns)} control-failover tests passed")
