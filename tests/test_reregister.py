"""
test_reregister.py — node re-register on coordinator restart (the harness enabler).

A coordinator restart leaves an empty topology; the node's heartbeat must detect that
(heartbeat → False) and re-register for its ALREADY-LOADED slot (prefer_range) so it keeps
serving the same layers instead of being handed a different range. Covers: heartbeat
known/unknown bool, prefer_range slot-stability across a fresh topology, and fallback when
the preferred range isn't a real slot.

Pure logic, no GPUs:  python3 -m tests.test_reregister
"""

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from engine.topology import Topology, Node  # noqa: E402

FP = "m@1"


def _node(nid, cap=32):
    return Node(node_id=nid, endpoint=("h", 1), capacity_layers=cap, model_fp=FP, payout_wallet="w")


def _mesh():
    return Topology(num_layers=64, coordinator_end=16, num_stages=3, model_fp=FP, replication=1)


def _range_of(t, nid):
    s = t.slots[t.nodes[nid].slot]
    return (s.start, s.end)


def main():
    t = _mesh()
    assert [(s.start, s.end) for s in t.slots] == [(16, 32), (32, 48), (48, 64)], "slot ranges"

    for nid in ("a", "b", "c"):
        t.register(_node(nid), now=1.0)
        t.mark_ready(nid)
        t.heartbeat(nid, 1.0)
    assign = {nid: _range_of(t, nid) for nid in ("a", "b", "c")}
    assert t.coverage_ok() and sorted(assign.values()) == [(16, 32), (32, 48), (48, 64)]

    # ── heartbeat known/unknown bool (the re-register signal) ─────────────────
    assert t.heartbeat("a", 2.0) is True, "known node → True"
    assert t.heartbeat("ghost", 2.0) is False, "unknown node → False (re-register signal)"

    # ── coordinator restart: fresh topology, nodes re-register for their loaded slot ──
    t2 = _mesh()
    for nid in ("a", "b", "c"):
        t2.register(_node(nid), now=1.0, prefer_range=assign[nid])   # ask for the SAME slot
        t2.mark_ready(nid)
        t2.heartbeat(nid, 1.0)
        assert _range_of(t2, nid) == assign[nid], f"{nid}: prefer_range {assign[nid]} not honored"
    assert t2.coverage_ok(), "re-registered mesh covered with STABLE slots (no stale-layer mismatch)"

    # ── prefer_range that isn't a real slot → falls back to a normal pick ──────
    t3 = _mesh()
    slot = t3.register(_node("x"), prefer_range=(99, 200))   # bogus range
    assert (slot.start, slot.end) in [(16, 32), (32, 48), (48, 64)], "bogus prefer_range falls back"
    t3.remove("x")
    assert t3.heartbeat("x", 5.0) is False, "removed node → heartbeat False"

    print("RE-REGISTER TESTS PASSED — heartbeat known/unknown bool, prefer_range slot-stability, fallback")


if __name__ == "__main__":
    main()
