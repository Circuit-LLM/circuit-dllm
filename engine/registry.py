"""
registry.py — the control service for the node mesh.

Wraps a `Topology` with the operator-facing control logic:

  - admission: allowlist/stake check + model-fingerprint, assign a slot, and issue
    the node its OWN derived data-wire key (retires the single shared cluster key —
    revoking one node never re-keys the rest).
  - liveness: heartbeat, drain, and a periodic `tick` (reap stably-dead nodes,
    surface re-balance needs + coverage status).
  - attribution: record which nodes served which layers per request, split the CIRC
    paid ∝ layers·tokens (minus a protocol fee), accrue per node, and settle on-chain
    in batches (never per-request — gas would dwarf the payment).

Pure logic + stdlib only (hmac/hashlib): unit-testable without GPUs or a chain. The
HTTP control channel and the coordinator call into this.
"""

from __future__ import annotations

import hashlib
import hmac
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from engine.topology import Topology, Node


def derive_node_key(master_secret: bytes, node_id: str) -> bytes:
    """Each node gets its own 32-byte data-wire key, derived from a master secret +
    its id. A node only ever holds its own key, and revoking one doesn't touch the
    rest — least privilege vs. the single shared cluster key."""
    return hmac.new(master_secret, node_id.encode(), hashlib.sha256).digest()


@dataclass
class Registry:
    topo: Topology
    master_secret: bytes
    coordinator_endpoint: Tuple                  # how a node reaches the coordinator
    allowlist: Optional[Set[str]] = None         # None = open; a set = permissioned
    fee_bps: int = 1000                          # protocol fee, basis points (10%)
    accrued: Dict[str, int] = field(default_factory=dict)  # node_id -> CIRC raw owed
    wallets: Dict[str, str] = field(default_factory=dict)  # node_id -> payout wallet
    # one lock serializes control-plane mutations (control-server threads) against
    # the coordinator reading a route snapshot (inference thread).
    _lock: object = field(default_factory=threading.RLock, repr=False, compare=False)

    # ── admission ─────────────────────────────────────────────────────────────
    def register(self, node: Node, now: float = 0.0) -> dict:
        with self._lock:
            if self.allowlist is not None and node.node_id not in self.allowlist:
                raise PermissionError(f"node {node.node_id} not on allowlist")
            slot = self.topo.register(node, now)   # raises on model mismatch / capacity
            key = derive_node_key(self.master_secret, node.node_id)
            node.wire_key = key                    # coordinator uses this to talk to the node
            self.wallets[node.node_id] = node.payout_wallet
            return {
                "assignment": {"start": slot.start, "end": slot.end},
                "model_fp": self.topo.model_fp,
                "session_key": key.hex(),
                "coordinator": self.coordinator_endpoint,
                "replication": self.topo.replication,
            }

    def heartbeat(self, node_id: str, now: float) -> None:
        with self._lock:
            self.topo.heartbeat(node_id, now)

    def drain(self, node_id: str) -> None:
        with self._lock:
            self.topo.drain(node_id)

    def mark_ready(self, node_id: str) -> None:
        """Node finished downloading its layers and is serving (JOINING -> READY)."""
        with self._lock:
            self.topo.mark_ready(node_id)

    def mark_suspect(self, node_id: str) -> None:
        with self._lock:
            self.topo.mark_suspect(node_id)

    def set_rtt(self, node_id: str, ms: float) -> None:
        """Record a measured coordinator→node RTT (ms). The active RTT prober calls
        this; proximity routing (topo.route_by_latency) then prefers closer holders."""
        with self._lock:
            self.topo.set_rtt(node_id, ms)

    def snapshot(self) -> dict:
        """JSON-able view of the live mesh for ops / dashboards / the /topology route."""
        with self._lock:
            return {
                "model_fp": self.topo.model_fp,
                "replication": self.topo.replication,
                "coverage_ok": self.topo.coverage_ok(),
                "slots": [
                    {"slot": s.index, "layers": [s.start, s.end],
                     "holders": [{"node_id": h, "state": self.topo.nodes[h].state,
                                  "region": self.topo.nodes[h].region,
                                  "rtt_ms": round(self.topo.rtt(h), 1)}
                                 for h in s.holders if h in self.topo.nodes]}
                    for s in self.topo.slots
                ],
            }

    def tick(self, now: float) -> dict:
        """Periodic maintenance: reap stably-dead nodes, surface re-balance needs +
        coverage status (for the operator / alerts / joiner placement)."""
        with self._lock:
            reaped = self.topo.reap(now)
            purged = self.topo.purge(now)
            targets = self.topo.rebalance_targets()
            return {
                "reaped": reaped,
                "purged": purged,
                "needs_holders": [{"slot": s.index, "layers": [s.start, s.end],
                                   "have": len(self.topo.holders(s.index)),
                                   "want": self.topo.replication} for s in targets],
                "coverage_ok": self.topo.coverage_ok(),
            }

    # ── routing (read) ────────────────────────────────────────────────────────
    def route_snapshot(self) -> List[Node]:
        """Thread-safe snapshot of the pipeline: the PRIMARY healthy holder for each
        slot, in order. The coordinator pins this for a session's lifetime (session
        affinity → warm KV). Raises if any slot is uncovered (the coverage invariant)."""
        with self._lock:
            return [holders[0] for _slot, holders in self.topo.route()]

    # ── attribution + payout ──────────────────────────────────────────────────
    def record_work(self, route: List[Tuple[str, int, int]], paid_raw: int) -> None:
        """route = [(node_id, num_layers, num_tokens), …] for one paid request. Split
        `paid_raw` CIRC (minus the protocol fee) across the servers ∝ layers·tokens
        and accrue to each. The coordinator is authoritative on the route, so this
        cannot be gamed by a node (that's a separate question from whether the work
        was *correct* — see JOIN.md §9)."""
        fee = paid_raw * self.fee_bps // 10_000
        pool = paid_raw - fee
        weights = [(nid, layers * tokens) for nid, layers, tokens in route]
        total = sum(w for _, w in weights)
        if total <= 0:
            return
        with self._lock:
            assigned = 0
            for i, (nid, w) in enumerate(weights):
                # integer split; the last server gets the rounding remainder so the
                # pool is conserved exactly (no lost lamports)
                share = (pool * w // total) if i < len(weights) - 1 else (pool - assigned)
                self.accrued[nid] = self.accrued.get(nid, 0) + share
                assigned += share

    def settle(self, min_payout_raw: int) -> List[Tuple[str, int]]:
        """Return the batch of (wallet, amount_raw) for nodes whose accrued balance
        is >= min_payout_raw, zeroing those. The caller does the on-chain Token-2022
        transfers. Below-threshold balances roll over (avoids dust-gas churn)."""
        with self._lock:
            batch = []
            for nid, amount in list(self.accrued.items()):
                if amount >= min_payout_raw and self.wallets.get(nid):
                    batch.append((self.wallets[nid], amount))
                    self.accrued[nid] = 0
            return batch
