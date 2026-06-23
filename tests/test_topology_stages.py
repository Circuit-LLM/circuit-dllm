"""
test_topology_stages.py — fewest-fattest stage planning (SPEED_ROADMAP §1.2).

Covers: plan_stages() chooses the smallest stage count the fleet can staff (fewer
stages = fewer hops), the 72B fat-node case, exact/remainder splits, capacity and
node-count infeasibility, heterogeneous fleets, replication, and the Topology.for_fleet
convenience that wires the planner into slot construction.

Pure logic, no GPUs:  python3 -m tests.test_topology_stages
"""

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from engine.topology import Topology, plan_stages  # noqa: E402

FP = "m@1"


def main():
    # ── the 72B fat-node case: 64 split layers, 24GB-class nodes hold ~40 each ──
    # k=1 needs a node ≥64 (none); k=2 → two 32-layer slots, both fit a 40-cap node.
    assert plan_stages(64, [40, 40, 40], replication=1) == 2, "72B → 2 fat stages (1 hop)"
    # thin 20-cap nodes can't hold 32 → must split finer: ceil-ish up to fit 20.
    # k=2→32>20, k=3→ slots 21,21,22 (22>20), k=4→16,16,16,16 all ≤20 → 4 stages.
    assert plan_stages(64, [20] * 4, replication=1) == 4, "thin nodes → more, smaller stages"

    # ── fewer/fatter is preferred: same model, fatter nodes ⇒ fewer hops ────────
    assert plan_stages(64, [64]) == 1, "one huge node holds the whole split → 0 inter-node hops"
    assert plan_stages(64, [33, 33]) == 2, "just-big-enough for halves → 2 stages"
    assert plan_stages(64, [32, 32]) == 2, "exactly half each → 2 stages"
    # 31-cap can't hold a 32 half, so it needs 3 stages (max slot = 22 ≤ 31).
    assert plan_stages(64, [31, 31, 31]) == 3, "just-too-small for halves → 3 stages"

    # ── remainder slot is the binding one (constructor puts remainder on the last) ─
    # budget 10, k=3 → slots [3,3,4]; the 4-slot needs a node ≥4.
    assert plan_stages(10, [4, 3, 3]) == 3, "remainder slot (4) needs the one 4-cap node"
    assert plan_stages(10, [3, 3, 3, 3]) == 4, "all 3-cap → 4 stages (slots ≤3: 3,3,2,2)"

    # ── replication multiplies the node requirement (k·rep assignable nodes) ────
    assert plan_stages(64, [40, 40, 40, 40], replication=2) == 2, "2 stages × 2 replicas = 4 nodes"
    try:
        plan_stages(64, [40, 40, 40], replication=2)  # only 3 nodes, need 4
        raise AssertionError("should raise: too few nodes for replication=2 at k=2")
    except ValueError:
        pass

    # ── infeasible: no node big enough even at the thinnest useful split ────────
    # capacity 1 can only ever hold 1-layer slots → needs k = layer_budget stages;
    # with only 2 nodes of cap 1, 64 layers can't be covered.
    try:
        plan_stages(64, [1, 1])
        raise AssertionError("should raise: capacities far too small")
    except ValueError:
        pass
    # the smallest-possible split still works if there are enough tiny nodes:
    assert plan_stages(3, [1, 1, 1]) == 3, "3 one-cap nodes cover a 3-layer model as 3 stages"

    # ── heterogeneous fleet: largest slot matched to largest node ───────────────
    # budget 64, k=2 slots [32,32]; nodes [50,32] → both halves fit (50≥32, 32≥32).
    assert plan_stages(64, [50, 32]) == 2, "heterogeneous: big+exact covers the two halves"
    # nodes [50,31]: the 31 can't hold a 32 half → fall to k=3 (slots 21,21,22; 50≥22,
    # 31≥21, and a 3rd node would be needed) — but only 2 nodes, so k=3 needs 3 → infeasible,
    # k grows until a 2-node split fits: impossible (every k≥2 needs k nodes ≥ the slots).
    # With 2 nodes the model must be 2 stages or fewer → here it can't, so it raises.
    try:
        plan_stages(64, [50, 31])
        raise AssertionError("should raise: 2 nodes but the smaller can't hold a half")
    except ValueError:
        pass

    # ── argument validation ─────────────────────────────────────────────────────
    for bad in (lambda: plan_stages(0, [10]),
                lambda: plan_stages(10, [5], replication=0)):
        try:
            bad(); raise AssertionError("should raise on bad args")
        except ValueError:
            pass

    # ── Topology.for_fleet wires the planner into real slot construction ────────
    # 72B-shaped: 80 layers, coordinator holds 0:16 → 64 split; fleet of 40-cap nodes.
    t = Topology.for_fleet(num_layers=80, coordinator_end=16, model_fp=FP,
                           capacities=[40, 40, 40], replication=1)
    assert len(t.slots) == 2, "for_fleet → 2 fat stages"
    assert (t.slots[0].start, t.slots[0].end) == (16, 48), "first fat slot 16:48"
    assert (t.slots[1].start, t.slots[1].end) == (48, 80), "second fat slot 48:80"
    # every slot is holdable by a 40-cap node
    assert all((s.end - s.start) <= 40 for s in t.slots), "all slots fit a 40-cap node"

    # for_fleet with thin nodes makes more stages but still a valid covering topology
    t2 = Topology.for_fleet(num_layers=80, coordinator_end=16, model_fp=FP,
                            capacities=[20] * 4, replication=1)
    assert len(t2.slots) == 4, "thin fleet → 4 stages"
    assert t2.slots[0].start == 16 and t2.slots[-1].end == 80, "slots cover the whole split"
    assert all((s.end - s.start) <= 20 for s in t2.slots), "all slots fit a 20-cap node"

    print("TOPOLOGY STAGES TESTS PASSED — fewest-fattest planning (72B fat-node = 2 stages), "
          "remainder/replication/infeasibility, heterogeneous fleets, for_fleet wiring")


if __name__ == "__main__":
    main()
