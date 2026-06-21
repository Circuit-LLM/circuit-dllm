"""
test_topology.py — the mesh source-of-truth.

Covers: slotting, model-fingerprint admission, capacity admission, replication-
aware assignment, the coverage invariant (incl. the chaos case where a whole slot
dies → routing must refuse), failover ordering, SUSPECT/recover, churn damping.

Pure logic, no GPUs:  python3 -m tests.test_topology
"""

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from engine.topology import Topology, Node, DEAD  # noqa: E402

FP = "qwen2.5-32b-awq@rev1"


def _node(nid, cap=16, fp=FP):
    return Node(node_id=nid, endpoint=("h", 1), capacity_layers=cap,
                model_fp=fp, payout_wallet="w")


def main():
    # 64-layer model, coordinator holds 0:32, two remote stages, R=2.
    t = Topology(num_layers=64, coordinator_end=32, num_stages=2,
                 model_fp=FP, replication=2, dead_after_s=30.0)
    assert [(s.start, s.end) for s in t.slots] == [(32, 48), (48, 64)], "slot ranges"

    # admission: model mismatch rejected
    try:
        t.register(_node("bad", fp="other-model"))
        assert False, "model mismatch should reject"
    except ValueError:
        pass

    # admission: capacity too small rejected (slot is 16 layers, node holds 8)
    try:
        t.register(_node("tiny", cap=8))
        assert False, "insufficient capacity should reject"
    except ValueError:
        pass

    # assign 4 nodes → 2 per slot (fills the emptiest slot first)
    for nid in ("n1", "n2", "n3", "n4"):
        t.register(_node(nid), now=100.0)
        t.mark_ready(nid)
        t.heartbeat(nid, 100.0)
    assert sorted(len(s.holders) for s in t.slots) == [2, 2], "balanced 2+2"
    assert t.coverage_ok() and t.under_replicated() == [], "both slots at R"
    assert len(t.route()) == 2, "valid pipeline"

    # failover ordering: the freshest-heartbeat holder is primary
    x, y = (n.node_id for n in t.holders(0))
    t.heartbeat(x, 150.0)
    t.heartbeat(y, 160.0)                       # y fresher → primary
    assert t.holders(0)[0].node_id == y, "primary = freshest heartbeat"

    # SUSPECT is not routed, still covered by the other holder, recovers on heartbeat
    t.mark_suspect(y)
    assert y not in [n.node_id for n in t.holders(0)], "suspect not routable"
    assert t.coverage_ok(), "slot0 still covered by x"
    assert any(s.index == 0 for s in t.under_replicated()), "slot0 now under R"
    t.heartbeat(y, 170.0)
    assert y in [n.node_id for n in t.holders(0)], "heartbeat clears suspicion"

    # ── one coherent clock for the death scenario ─────────────────────────────
    T = 1000.0
    for nid in ("n1", "n2", "n3", "n4"):       # everyone fresh at T
        t.heartbeat(nid, T)
    assert t.coverage_ok() and t.under_replicated() == []

    s0 = [n.node_id for n in t.holders(0)]
    v1, v2 = s0[0], s0[1]

    # churn damping: a brief silence does NOT kill
    t.nodes[v1].last_hb = T - 10               # 10s < 30s window
    assert t.reap(T) == [], "not reaped within the window"

    # one holder stably dead → still covered, slot0 now needs a replica
    t.nodes[v1].last_hb = T - 40               # past dead_after_s
    dead = t.reap(T)
    assert v1 in dead and t.nodes[v1].state == DEAD
    assert t.coverage_ok(), "surviving holder keeps slot0 covered"
    assert any(s.index == 0 for s in t.rebalance_targets()), "slot0 under-replicated"

    # CHAOS: the OTHER slot0 holder dies too → coverage gap → route() must refuse
    t.nodes[v2].last_hb = T - 40
    t.reap(T)
    assert not t.coverage_ok(), "slot0 fully dead → not covered"
    try:
        t.route()
        assert False, "route() must refuse a pipeline with a coverage gap"
    except RuntimeError as e:
        assert "coverage gap" in str(e)

    # recovery: a new node is assigned to the most-urgent (uncovered) slot first
    t.register(_node("n5"), now=T)
    t.mark_ready("n5")
    t.heartbeat("n5", T)
    assert t.nodes["n5"].slot == 0, "new node fills the uncovered slot"
    assert t.coverage_ok(), "coverage restored"

    print("TOPOLOGY TESTS PASSED — assignment, coverage invariant, failover, churn damping")


if __name__ == "__main__":
    main()
