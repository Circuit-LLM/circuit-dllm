"""
test_registry.py — the control service: admission, per-node keys, attribution
split (∝ layers·tokens, fee, conservation), batched settlement, liveness tick.

Pure logic, no GPUs / no chain:  python3 -m tests.test_registry
"""

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from engine.topology import Topology, Node  # noqa: E402
from engine.registry import Registry, derive_node_key  # noqa: E402

FP = "qwen2.5-32b-awq@rev1"


def _node(nid, cap=16, fp=FP, wallet="W"):
    return Node(node_id=nid, endpoint=("h", 1), capacity_layers=cap,
                model_fp=fp, payout_wallet=wallet)


def main():
    topo = Topology(num_layers=64, coordinator_end=32, num_stages=2,
                    model_fp=FP, replication=2, dead_after_s=30.0)
    reg = Registry(topo=topo, master_secret=b"master", coordinator_endpoint=("c", 9),
                   allowlist={"n1", "n2"}, fee_bps=1000)

    # admission: allowlist enforced
    try:
        reg.register(_node("stranger"))
        assert False, "non-allowlisted node should be rejected"
    except PermissionError:
        pass

    # admission: model fingerprint enforced (via topology)
    try:
        reg.register(_node("n1", fp="other"))
        assert False, "model mismatch should be rejected"
    except ValueError:
        pass

    # success: returns assignment + per-node key + model fp
    r1 = reg.register(_node("n1", wallet="WALLET1"), now=100.0)
    assert r1["assignment"] == {"start": 32, "end": 48}
    assert r1["model_fp"] == FP
    assert r1["session_key"] == derive_node_key(b"master", "n1").hex()
    assert r1["session_key"] != derive_node_key(b"master", "n2").hex(), "per-node keys differ"

    reg.register(_node("n2", wallet="WALLET2"), now=100.0)
    for nid in ("n1", "n2"):
        topo.mark_ready(nid)
        reg.heartbeat(nid, 100.0)

    # ── attribution: split a payment ∝ layers·tokens, minus fee, conserved ──────
    # two stages served: n1 did 16 layers, n2 did 16 layers, both over 10 tokens.
    paid = 1000  # CIRC raw
    reg.record_work(route=[("n1", 16, 10), ("n2", 16, 10)], paid_raw=paid)
    fee = paid * 1000 // 10_000          # 10% = 100
    pool = paid - fee                    # 900, split 50/50
    assert reg.accrued["n1"] == 450 and reg.accrued["n2"] == 450
    assert reg.accrued["n1"] + reg.accrued["n2"] == pool, "pool conserved"

    # uneven weights: n1 twice the work of n2 → 2:1 split of the (post-fee) pool
    reg.accrued.clear()
    reg.record_work(route=[("n1", 20, 10), ("n2", 10, 10)], paid_raw=300)
    pool2 = 300 - 30                     # 270; weights 200:100 → 180:90
    assert reg.accrued["n1"] == 180 and reg.accrued["n2"] == 90
    assert reg.accrued["n1"] + reg.accrued["n2"] == pool2, "conserved with remainder"

    # ── settlement: batch only >= threshold, zero those, roll over the rest ─────
    reg.accrued = {"n1": 5000, "n2": 100}   # n1 over threshold, n2 under
    batch = reg.settle(min_payout_raw=1000)
    assert batch == [("WALLET1", 5000)], "only n1 settles"
    assert reg.accrued["n1"] == 0 and reg.accrued["n2"] == 100, "n2 rolls over"

    # ── liveness tick: reap a stale node, surface the coverage gap ──────────────
    topo.nodes["n1"].last_hb = 100.0 - 40   # stale (we'll reap at 100)
    status = reg.tick(now=100.0)
    assert "n1" in status["reaped"]
    assert status["coverage_ok"] is False, "slot0 lost its only ready holder"
    assert any(h["slot"] == 0 for h in status["needs_holders"]), "slot0 needs a holder"

    print("REGISTRY TESTS PASSED — admission, per-node keys, attribution split, settlement, tick")


if __name__ == "__main__":
    main()
