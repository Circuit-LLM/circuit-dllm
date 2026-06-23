"""
test_topology_latency.py — topology-aware routing foundation (P1).

Covers: the region-distance RTT estimate, measured-RTT override, proximity holder
ordering (CIRCUIT_ROUTE_LATENCY), the default-path regression guard (flag off ==
heartbeat ordering, byte-identical), RTT/heartbeat tie-break, and rtt-map cleanup.

Pure logic, no GPUs:  python3 -m tests.test_topology_latency
"""

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from engine.topology import Topology, Node, _region_rtt_estimate, _DEFAULT_RTT_MS  # noqa: E402

FP = "m@1"


def _node(nid, cap=32, region=None):
    return Node(node_id=nid, endpoint=("h", 1), capacity_layers=cap,
                model_fp=FP, region=region, payout_wallet="w")


def main():
    # ── region-distance estimate: local < regional < intercontinental ─────────
    assert _region_rtt_estimate("na-east", "na-east") < _region_rtt_estimate("na-east", "na-west")
    assert _region_rtt_estimate("na-east", "na-west") < _region_rtt_estimate("na-east", "eu-west")
    assert _region_rtt_estimate(None, "na-east") == _DEFAULT_RTT_MS, "unknown → neutral default"
    assert _region_rtt_estimate("na-east", None) == _DEFAULT_RTT_MS

    # ── default path unchanged: flag OFF → primary is freshest heartbeat ───────
    t = Topology(num_layers=64, coordinator_end=32, num_stages=1, model_fp=FP, replication=3)
    for nid in ("a", "b", "c"):
        t.register(_node(nid), now=100.0)
        t.mark_ready(nid)
    t.heartbeat("a", 101.0); t.heartbeat("b", 103.0); t.heartbeat("c", 102.0)
    assert t.holders(0)[0].node_id == "b", "flag off → freshest heartbeat is primary"
    assert [n.node_id for n in t.holders(0)] == ["b", "c", "a"], "flag off → heartbeat order"

    # ── proximity routing: flag ON + measured RTTs → primary is lowest RTT ─────
    t.route_by_latency = True
    t.set_rtt("a", 200.0); t.set_rtt("b", 150.0); t.set_rtt("c", 20.0)
    assert [n.node_id for n in t.holders(0)] == ["c", "b", "a"], "ordered by RTT asc"
    assert t.route()[0][1][0].node_id == "c", "route() primary = lowest-RTT holder"

    # ── rtt() resolution: measured > region-estimate > default ────────────────
    t2 = Topology(num_layers=40, coordinator_end=20, num_stages=1, model_fp=FP,
                  replication=2, coordinator_region="na-east", route_by_latency=True)
    t2.register(_node("near", region="na-east"), now=0.0); t2.mark_ready("near")
    t2.register(_node("far", region="eu-west"), now=0.0); t2.mark_ready("far")
    t2.heartbeat("near", 1.0); t2.heartbeat("far", 1.0)
    assert t2.rtt("near") < t2.rtt("far"), "region estimate: same-region closer"
    assert t2.holders(0)[0].node_id == "near", "region estimate routes to same-region holder"

    # a measured probe supersedes the region estimate (e.g. the near node is congested)
    t2.set_rtt("far", 2.0)
    assert t2.rtt("far") == 2.0 and t2.holders(0)[0].node_id == "far", "measured RTT wins"

    # ── tie-break: equal RTT → freshest heartbeat ─────────────────────────────
    t2.set_rtt("near", 2.0); t2.set_rtt("far", 2.0)
    t2.heartbeat("near", 10.0); t2.heartbeat("far", 20.0)
    assert t2.holders(0)[0].node_id == "far", "equal RTT → freshest heartbeat breaks tie"

    # ── churn: remove() drops the rtt entry (no leak) ─────────────────────────
    t2.remove("far")
    assert "far" not in t2._rtt, "remove() cleans the rtt map"

    print("TOPOLOGY LATENCY TESTS PASSED — region estimate, measured override, "
          "proximity routing, tie-break, default-path regression, cleanup")


if __name__ == "__main__":
    main()
