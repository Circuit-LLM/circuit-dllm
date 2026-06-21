"""
test_churn.py — node leave-and-recover lifecycle on the control plane (no GPUs).

Exercises the full churn cycle the failover tests don't:
  - a node goes silent; the reaper marks it DEAD only AFTER the damping window
    (not on the first missed beat) — churn damping;
  - its slot is surfaced as under-replicated, but coverage NEVER breaks (the replica
    keeps serving) — the coverage invariant holds throughout;
  - a replacement joiner is PLACED on the needy slot, restoring R — self-healing
    placement (passive: recovery happens when a node joins; nothing auto-recruits).

Pure topology/registry logic; runs anywhere:  python3 -m tests.test_churn
"""

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from engine.topology import Topology, Node, DEAD  # noqa: E402
from engine.registry import Registry  # noqa: E402

FP = "qwen2.5-32b-awq@rev1"


def mk(nid, cap=12):
    return Node(node_id=nid, endpoint=("10.0.0.1", 9000), capacity_layers=cap, model_fp=FP)


def main():
    DEAD_AFTER = 30.0
    topo = Topology(num_layers=24, coordinator_end=0, num_stages=2,
                    model_fp=FP, replication=2, dead_after_s=DEAD_AFTER)
    reg = Registry(topo=topo, master_secret=b"secret", coordinator_endpoint=("c", 0))
    T = 1000.0

    # 4 nodes join → 2 per slot → all READY. Full R=2, fully covered.
    for nid in ("a0", "a1", "b0", "b1"):
        reg.register(mk(nid), T)
        reg.mark_ready(nid)
    assert topo.coverage_ok(), "fully covered after join"
    assert not topo.under_replicated(), "R=2 on every slot"
    slot_of = {n.node_id: n.slot for n in topo.nodes.values()}
    s0 = sorted(nid for nid in ("a0", "a1", "b0", "b1") if slot_of[nid] == 0)
    assert len(s0) == 2, f"two holders on slot 0: {slot_of}"
    victim, survivor = s0[0], s0[1]
    print(f"joined: 4 nodes, R=2, covered. victim={victim} survivor={survivor} (slot 0)")

    alive = ["a0", "a1", "b0", "b1"]

    def beat(t):                       # everyone alive heartbeats except the victim
        for nid in alive:
            if nid != victim:
                reg.heartbeat(nid, t)

    # 1) within the damping window — a tick must NOT reap the victim yet
    beat(T + DEAD_AFTER - 1)
    st = reg.tick(T + DEAD_AFTER - 1)
    assert topo.nodes[victim].state != DEAD, "victim reaped too early (no churn damping)"
    assert st["coverage_ok"] and not st["reaped"], st
    print(f"  t+{DEAD_AFTER-1:.0f}: victim silent but NOT reaped (damping holds); covered")

    # 2) past the window — the victim is reaped; slot 0 drops to R=1 but stays COVERED
    beat(T + DEAD_AFTER + 1)
    st = reg.tick(T + DEAD_AFTER + 1)
    assert topo.nodes[victim].state == DEAD, f"victim should be DEAD: {topo.nodes[victim].state}"
    assert st["coverage_ok"], "coverage must hold — the survivor still serves slot 0"
    under = {s.index for s in topo.under_replicated()}
    assert under == {0}, f"slot 0 under-replicated, slot 1 fine — got {under}"
    assert topo.holders(0) and topo.holders(0)[0].node_id == survivor, "survivor serves slot 0"
    print(f"  t+{DEAD_AFTER+1:.0f}: victim REAPED; slot 0 at R=1 (under-replicated) but still covered")

    # 3) a replacement joins → placed on the NEEDY slot 0 → R restored, no gap ever
    reg.register(mk("a2"), T + DEAD_AFTER + 2)
    reg.mark_ready("a2")
    alive.append("a2")
    assert topo.nodes["a2"].slot == 0, f"new node must heal slot 0, got slot {topo.nodes['a2'].slot}"
    assert not topo.under_replicated(), "R=2 restored on every slot"
    assert topo.coverage_ok(), "still fully covered"
    print("  replacement a2 placed on slot 0 → R=2 restored, coverage intact throughout")

    # 4) the reaped node stays visible (DEAD) for one observability window, then is
    #    PURGED so holder lists / the node table don't grow without bound under churn.
    assert victim in topo.nodes and topo.nodes[victim].state == DEAD, "DEAD but still visible"
    later = T + 2 * DEAD_AFTER + 2
    for nid in alive:                      # survivors keep beating; victim stays silent
        if nid != victim:
            reg.heartbeat(nid, later)
    st = reg.tick(later)
    assert victim in st["purged"], f"victim should be purged: {st['purged']}"
    assert victim not in topo.nodes, "purged node removed from the node table"
    assert victim not in topo.slots[0].holders, "purged node removed from the holder list"
    assert topo.coverage_ok() and not topo.under_replicated(), "still healthy after purge"
    print(f"  t+{2*DEAD_AFTER+2:.0f}: victim PURGED (removed from nodes + holders); mesh healthy")

    print("CHURN LIFECYCLE PASSED — damping, coverage invariant, self-healing "
          "placement, and bounded-memory purge all hold across leave-and-recover")


if __name__ == "__main__":
    main()
