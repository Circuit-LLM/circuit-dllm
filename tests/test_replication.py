"""
test_replication.py — load-balanced session routing across slot replicas (aggregate scale).

Registry.acquire_route spreads concurrent sessions over each slot's replicas so R replicas yield
~R parallel pipelines; release_route frees the load; replication=1 is identical to the primary
(route_snapshot). Pure logic, no GPUs/sockets.

    python3 -m tests.test_replication
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from engine.topology import Topology, Node  # noqa: E402
from engine.registry import Registry  # noqa: E402

FP = "m@1"


def _node(nid, cap=4):
    return Node(node_id=nid, endpoint=("h", 1), capacity_layers=cap, model_fp=FP, payout_wallet="w")


def _reg(topo):
    return Registry(topo=topo, master_secret=b"s" * 32, coordinator_endpoint=("c", 1))


def main():
    # ── 2 slots × 2 replicas: sessions spread into disjoint parallel lanes ─────
    topo = Topology(num_layers=8, coordinator_end=0, num_stages=2, model_fp=FP, replication=2)
    reg = _reg(topo)
    for nid in ["a", "b", "c", "d"]:
        reg.register(_node(nid), now=1.0); topo.mark_ready(nid)
    assert all(len(s.holders) == 2 for s in topo.slots), "4 nodes → 2 holders per slot"

    rA = [n.node_id for n in reg.acquire_route("A")]
    rB = [n.node_id for n in reg.acquire_route("B")]
    assert len(rA) == 2 and len(rB) == 2, "route covers both slots"
    assert set(rA).isdisjoint(rB), f"two sessions → DISJOINT replica lanes (parallel): {rA} vs {rB}"
    assert sorted(reg.load_snapshot().values()) == [1, 1, 1, 1], "every holder used once (balanced)"

    rC = [n.node_id for n in reg.acquire_route("C")]
    assert max(reg.load_snapshot().values()) == 2, "3rd session reuses a lane (load 2 on a slot)"

    # release frees load
    reg.release_route("A")
    assert sum(reg.load_snapshot().values()) == 4, "after releasing A: B(2)+C(2) holders remain"
    reg.release_route("B"); reg.release_route("C")
    assert sum(reg.load_snapshot().values()) == 0, "all released → zero load"
    reg.release_route("nope")  # idempotent: unknown session is a no-op

    # ── re-acquire for the SAME session releases its prior route (no double-count) ─
    reg.acquire_route("S"); reg.acquire_route("S")
    assert sum(reg.load_snapshot().values()) == 2, "re-acquire S → still just 2 holders, not 4"
    reg.release_route("S")

    # ── replication=1: balanced == primary (safe default, no behavior change) ──
    t1 = Topology(num_layers=8, coordinator_end=0, num_stages=2, model_fp=FP, replication=1)
    r1 = _reg(t1)
    for nid in ["x", "y"]:
        r1.register(_node(nid), now=1.0); t1.mark_ready(nid)
    assert [n.node_id for n in r1.route_snapshot()] == [n.node_id for n in r1.acquire_route("S1")], \
        "replication=1 → acquire_route picks the same holders as route_snapshot"

    # ── coverage gap → acquire raises (same invariant as route) ────────────────
    t2 = Topology(num_layers=8, coordinator_end=0, num_stages=2, model_fp=FP, replication=1)
    r2 = _reg(t2)
    r2.register(_node("z"), now=1.0); t2.mark_ready("z")   # only slot 0 covered
    try:
        r2.acquire_route("G"); raise AssertionError("should raise on uncovered slot")
    except RuntimeError:
        pass

    print("REPLICATION TESTS PASSED — disjoint parallel lanes across replicas, balanced load, "
          "release frees load, re-acquire no double-count, replication=1 == primary, coverage gap raises")


if __name__ == "__main__":
    main()
