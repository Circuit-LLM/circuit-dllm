"""
test_control_concurrency.py — control-plane robustness: drain + concurrency.

Two CPU-only checks that complete the reliability story for real, churning meshes:

  DRAIN — a node told to drain is immediately excluded from NEW routing (so it can
  finish in-flight work and leave), while its replica keeps the slot covered; draining
  the only holder of a slot surfaces a coverage gap (never a silent hole).

  CONCURRENCY — the inference thread reads route snapshots while many control threads
  register / heartbeat / mark-ready / tick concurrently. The registry's lock must keep
  this consistent: no exception, no lost node, coverage intact at the end.

Pure topology/registry logic; runs anywhere:  python3 -m tests.test_control_concurrency
"""

import os
import sys
import threading

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from engine.topology import Topology, Node, DRAINING  # noqa: E402
from engine.registry import Registry  # noqa: E402

FP = "qwen2.5-32b-awq@rev1"


def mk(nid, cap=12):
    return Node(node_id=nid, endpoint=("10.0.0.1", 9000), capacity_layers=cap, model_fp=FP)


def test_drain():
    topo = Topology(num_layers=24, coordinator_end=0, num_stages=2,
                    model_fp=FP, replication=2, dead_after_s=1e18)
    reg = Registry(topo=topo, master_secret=b"s", coordinator_endpoint=("c", 0))
    for nid in ("a0", "a1", "b0", "b1"):
        reg.register(mk(nid), 1.0)
        reg.mark_ready(nid)
    s0 = sorted(n for n, nd in topo.nodes.items() if nd.slot == 0)
    # drain one slot-0 holder -> excluded from routing, replica still covers the slot
    reg.drain(s0[0])
    assert topo.nodes[s0[0]].state == DRAINING
    routed = [n.node_id for n in reg.route_snapshot()]
    assert s0[0] not in routed, "a draining node must not be routed to"
    assert s0[1] in routed and topo.coverage_ok(), "the replica keeps slot 0 covered"
    # drain the replica too -> slot 0 has no routable holder -> coverage gap surfaced
    reg.drain(s0[1])
    assert not topo.coverage_ok(), "draining all holders of a slot -> coverage gap"
    try:
        reg.route_snapshot()
        raise AssertionError("route_snapshot must refuse a routing with a hole")
    except RuntimeError as e:
        assert "coverage gap" in str(e)
    print("  DRAIN ok — draining excludes from routing, replica covers, all-drained = gap")


def test_concurrency():
    topo = Topology(num_layers=24, coordinator_end=0, num_stages=2,
                    model_fp=FP, replication=8, dead_after_s=1e18)
    reg = Registry(topo=topo, master_secret=b"s", coordinator_endpoint=("c", 0))
    # seed one ready holder per slot so route_snapshot is valid throughout
    for nid in ("seed0", "seed1"):
        reg.register(mk(nid), 0.0)
        reg.mark_ready(nid)
    errors = []
    N = 12

    def churn(i):
        try:
            nid = f"n{i}"
            reg.register(mk(nid), float(i))
            for t in range(20):
                reg.heartbeat(nid, float(i + t))
                reg.mark_ready(nid)
                reg.tick(float(1000 + t))
        except Exception as e:   # noqa: BLE001
            errors.append(repr(e))

    def reader():
        try:
            for _ in range(200):
                reg.route_snapshot()       # the inference thread's hot read
                reg.snapshot()
        except Exception as e:   # noqa: BLE001
            errors.append(repr(e))

    threads = [threading.Thread(target=churn, args=(i,)) for i in range(N)] + \
              [threading.Thread(target=reader) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent control-plane ops raced: {errors[:3]}"
    assert len(topo.nodes) == N + 2, f"all {N + 2} nodes present, none lost: {len(topo.nodes)}"
    assert topo.coverage_ok(), "still fully covered after concurrent churn"
    print(f"  CONCURRENCY ok — {N} churn threads + 3 readers, no race, "
          f"{len(topo.nodes)} nodes intact, covered")


def main():
    test_drain()
    test_concurrency()
    print("CONTROL-PLANE ROBUSTNESS PASSED — drain + concurrent access both consistent")


if __name__ == "__main__":
    main()
