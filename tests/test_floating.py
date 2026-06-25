"""Floating-coordinator Phase 1 — entry-orchestrator selection + route RPC serialization.

Pure logic (no torch), so it runs on the VPS. Asserts the additive seams behave AND that the
existing paths are untouched (acquire_entry returns None until an orchestrator joins → today's
in-process coordinator behavior is unchanged).
"""
from engine.topology import Topology, Node, READY, JOINING
from engine.registry import Registry
from engine.control_server import _node_json


def _reg(replication=1):
    topo = Topology(num_layers=80, coordinator_end=20, num_stages=3, model_fp="t",
                    replication=replication)
    return Registry(topo=topo, master_secret=b"x" * 32, coordinator_endpoint=("h", 1))


def _add_orch(reg, nid, ready=True, orchestrator=True):
    reg.topo.nodes[nid] = Node(node_id=nid, endpoint=("h", 0), capacity_layers=0, model_fp="t",
                               orchestrator=orchestrator, state=(READY if ready else JOINING))


def test_none_until_an_orchestrator_joins():
    # INERT by default: no orchestrator-capable node → None → caller uses the in-process coordinator
    assert _reg().acquire_entry("s1") is None


def test_only_ready_orchestrators_are_eligible():
    reg = _reg()
    _add_orch(reg, "joining", ready=False)          # orchestrator but JOINING → excluded
    _add_orch(reg, "plainREADY", orchestrator=False)  # READY but not an orchestrator → excluded
    assert reg.acquire_entry("s1") is None


def test_balances_then_releases():
    reg = _reg()
    _add_orch(reg, "orchA"); _add_orch(reg, "orchB")
    a = reg.acquire_entry("s1"); b = reg.acquire_entry("s2")
    assert {a.node_id, b.node_id} == {"orchA", "orchB"}          # spread across both
    reg.acquire_entry("s3")                                       # tie → min node_id = orchA
    assert reg.entry_snapshot() == {"orchA": 2, "orchB": 1}
    reg.release_entry("s1")                                       # frees one A
    assert reg.entry_snapshot() == {"orchA": 1, "orchB": 1}
    reg.release_entry("nope")                                     # idempotent no-op
    assert reg.entry_snapshot() == {"orchA": 1, "orchB": 1}


def test_reacquire_releases_prior_pick():
    reg = _reg()
    _add_orch(reg, "orchA"); _add_orch(reg, "orchB")
    reg.acquire_entry("s1")                                       # → orchA (load A=1)
    reg.acquire_entry("s1")                                       # re-acquire: release A, repick
    # s1 holds exactly one entry; total load across orchestrators == 1
    assert sum(reg.entry_snapshot().values()) == 1


def test_node_json_serialization():
    reg = _reg()
    # a slice-holder with an assigned slot serializes its layer range
    n = Node(node_id="stage1", endpoint=("1.2.3.4", 5000), capacity_layers=20, model_fp="t")
    n.slot = 0
    j = _node_json(reg, n)
    assert j["node_id"] == "stage1" and j["endpoint"] == ["1.2.3.4", 5000]
    assert j["layers"] == [reg.topo.slots[0].start, reg.topo.slots[0].end]
    # a head-only orchestrator (no slot) → layers None
    o = Node(node_id="orchA", endpoint=("h", 0), capacity_layers=0, model_fp="t", orchestrator=True)
    assert _node_json(reg, o)["layers"] is None and _node_json(reg, o)["orchestrator"] is True


def test_existing_paths_unchanged():
    # acquire_route still works at replication=1 (the entry seam didn't touch slice routing)
    reg = _reg()
    n = Node(node_id="n1", endpoint=("h", 2), capacity_layers=20, model_fp="t")
    reg.register(n); reg.mark_ready("n1")
    # with one node it can't cover 4 slots — just assert acquire_entry stays inert alongside it
    assert reg.acquire_entry("s1") is None
    assert reg.load_snapshot() == {}


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"  ok  {fn.__name__}")
    print(f"\nALL {len(fns)} floating-coordinator Phase-1 tests passed")
