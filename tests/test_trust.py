"""
test_trust.py — trustless node verification (docs/VERIFICATION.md).

Covers the control-plane half (pure logic, no torch / no mesh):
  - new nodes are PROBATION; the seed fleet is admitted TRUSTED
  - holders() orders TRUSTED-first → a probation node is never the primary while a trusted
    holder exists, but still counts for coverage when it's the only holder
  - record_check: promote after N passes, a pass forgives a strike, evict after M fails,
    and even a TRUSTED node can be demoted+evicted (trust isn't permanent)
  - audit_pairs() yields (probation, trusted-reference, slot) only where both exist

Run:  python3 -m tests.test_trust
"""

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from engine.topology import Topology, Node, PROBATION, TRUSTED, DEAD, READY  # noqa: E402
from engine.registry import Registry  # noqa: E402

FP = "qwen2.5-test"


def _node(nid, cap=8):
    return Node(node_id=nid, endpoint=("h", 1), capacity_layers=cap, model_fp=FP, payout_wallet="w")


def _ready(topo, nid, t=0.0):
    topo.mark_ready(nid)
    topo.heartbeat(nid, t)


def test_admission_default_probation_seed_trusted():
    topo = Topology(8, coordinator_end=2, num_stages=2, model_fp=FP, replication=2)
    reg = Registry(topo=topo, master_secret=b"x" * 32, coordinator_endpoint=("c", 1),
                   seed_nodes={"seed1"})
    reg.register(_node("seed1"), now=0.0)
    reg.register(_node("newbie"), now=0.0)
    assert topo.nodes["seed1"].trust == TRUSTED, "seed node admitted trusted"
    assert topo.nodes["newbie"].trust == PROBATION, "non-seed node starts on probation"


def test_holders_trusted_first_but_probation_covers():
    topo = Topology(8, coordinator_end=2, num_stages=1, model_fp=FP, replication=3)
    for nid in ("p1", "t1", "p2"):
        topo.register(_node(nid), now=0.0)
        _ready(topo, nid)
    topo.set_trusted("t1")
    holders = topo.holders(0)
    assert holders[0].node_id == "t1", "trusted holder is primary regardless of join order"
    assert {h.node_id for h in holders} == {"p1", "t1", "p2"}, "probation holders still routable (coverage)"

    # a slot with ONLY probation holders still routes (coverage beats purity) — but logged elsewhere
    topo2 = Topology(8, coordinator_end=2, num_stages=1, model_fp=FP, replication=2)
    for nid in ("a", "b"):
        topo2.register(_node(nid), now=0.0); _ready(topo2, nid)
    assert topo2.coverage_ok(), "all-probation slot is still covered"
    assert topo2.holders(0), "route() can fall back to a probation primary"


def test_record_check_promote_forgive_evict():
    topo = Topology(8, coordinator_end=2, num_stages=1, model_fp=FP, replication=2)
    topo.register(_node("n"), now=0.0); _ready(topo, "n")

    assert topo.record_check("n", True, promote_after=3) is None
    assert topo.record_check("n", True, promote_after=3) is None
    assert topo.nodes["n"].trust == PROBATION, "not promoted before promote_after passes"
    assert topo.record_check("n", True, promote_after=3) == "promoted"
    assert topo.nodes["n"].trust == TRUSTED, "promoted on the 3rd pass"

    # a fresh node: a pass forgives a prior strike (transient blip shouldn't doom it)
    topo.register(_node("m"), now=0.0); _ready(topo, "m")
    topo.record_check("m", False, strike_max=2)
    assert topo.nodes["m"].strikes == 1
    topo.record_check("m", True, promote_after=99)
    assert topo.nodes["m"].strikes == 0, "a pass forgives one strike"

    # eviction: strike_max fails → DEAD, leaves trust + routing
    topo.register(_node("bad"), now=0.0); _ready(topo, "bad")
    assert topo.record_check("bad", False, strike_max=2) is None
    assert topo.record_check("bad", False, strike_max=2) == "evicted"
    assert topo.nodes["bad"].state == DEAD, "evicted node is DEAD"
    assert all(h.node_id != "bad" for h in topo.holders(0)), "evicted node no longer routable"


def test_trusted_node_can_be_evicted():
    topo = Topology(8, coordinator_end=2, num_stages=1, model_fp=FP, replication=2)
    topo.register(_node("t"), now=0.0); _ready(topo, "t"); topo.set_trusted("t")
    topo.record_check("t", False, strike_max=2)
    assert topo.record_check("t", False, strike_max=2) == "evicted", "trust isn't permanent"
    assert topo.nodes["t"].state == DEAD


def test_audit_pairs():
    topo = Topology(8, coordinator_end=2, num_stages=1, model_fp=FP, replication=3)
    reg = Registry(topo=topo, master_secret=b"x" * 32, coordinator_endpoint=("c", 1),
                   seed_nodes={"ref"})
    for nid in ("ref", "p1", "p2"):
        reg.register(_node(nid), now=0.0); _ready(topo, nid)
    pairs = reg.audit_pairs()
    assert {(p, r) for p, r, _ in pairs} == {("p1", "ref"), ("p2", "ref")}, \
        "each probation node pairs with the trusted reference of its slot"

    # no trusted holder on a slot → no audit pair (can't verify without a reference)
    topo2 = Topology(8, coordinator_end=2, num_stages=1, model_fp=FP, replication=2)
    reg2 = Registry(topo=topo2, master_secret=b"x" * 32, coordinator_endpoint=("c", 1))
    for nid in ("a", "b"):
        reg2.register(_node(nid), now=0.0); _ready(topo2, nid)
    assert reg2.audit_pairs() == [], "no trusted reference → nothing auditable"


def main():
    test_admission_default_probation_seed_trusted()
    test_holders_trusted_first_but_probation_covers()
    test_record_check_promote_forgive_evict()
    test_trusted_node_can_be_evicted()
    test_audit_pairs()
    print("TRUST TESTS PASSED — probation/seed admission, trusted-first routing, "
          "promote/forgive/evict, audit pairing")


if __name__ == "__main__":
    main()
