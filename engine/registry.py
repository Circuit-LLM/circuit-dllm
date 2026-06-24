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

from engine.topology import Topology, Node, PROBATION, TRUSTED, READY


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
    coordinator_wallet: str = ""                 # the coordinator's OWN payout wallet (it serves the co-located
                                                 # layer slice [0, CIRCUIT_COORD_LAYERS); "" = hub not paid out)
    allowlist: Optional[Set[str]] = None         # None = open (default); a set = frozen kill-switch
    seed_nodes: Optional[Set[str]] = None        # bootstrap fleet → TRUSTED on register (verify ref)
    fee_bps: int = 1000                          # protocol fee, basis points (10%)
    accrued: Dict[str, int] = field(default_factory=dict)  # node_id -> CIRC raw owed
    wallets: Dict[str, str] = field(default_factory=dict)  # node_id -> payout wallet
    # one lock serializes control-plane mutations (control-server threads) against
    # the coordinator reading a route snapshot (inference thread).
    _lock: object = field(default_factory=threading.RLock, repr=False, compare=False)
    # replication load-balancing: active-session count per holder + each session's chosen route,
    # so concurrent sessions spread across a slot's replicas (R replicas → ~R parallel pipelines).
    _load: Dict[str, int] = field(default_factory=dict, repr=False, compare=False)
    _session_load: Dict[object, list] = field(default_factory=dict, repr=False, compare=False)

    # ── admission ─────────────────────────────────────────────────────────────
    def register(self, node: Node, now: float = 0.0, prefer_range=None) -> dict:
        with self._lock:
            if self.allowlist is not None and node.node_id not in self.allowlist:
                raise PermissionError(f"node {node.node_id} not on allowlist")
            # prefer_range: a re-registering node asks for its already-loaded slot back.
            slot = self.topo.register(node, now, prefer_range=prefer_range)   # raises on mismatch / capacity
            # Bootstrap fleet is trusted outright so newcomers have a reference to be verified
            # against; everyone else stays PROBATION (default) until they pass challenges.
            if self.seed_nodes and node.node_id in self.seed_nodes:
                self.topo.set_trusted(node.node_id)
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

    def heartbeat(self, node_id: str, now: float) -> bool:
        """Returns False if the node is unknown (coordinator restarted) → it re-registers."""
        with self._lock:
            return self.topo.heartbeat(node_id, now)

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

    # ── trust / verification (docs/VERIFICATION.md) ────────────────────────────
    def set_trusted(self, node_id: str) -> None:
        with self._lock:
            self.topo.set_trusted(node_id)

    def record_check(self, node_id: str, ok: bool, promote_after: int = 3,
                     strike_max: int = 2) -> Optional[str]:
        """Fold one verification result into a node's trust (promote/evict). Returns
        'promoted' | 'evicted' | None. Called by the auditor after each challenge."""
        with self._lock:
            return self.topo.record_check(node_id, ok, promote_after=promote_after,
                                          strike_max=strike_max)

    def audit_pairs(self) -> List[Tuple[str, str, int]]:
        """(probation_node, trusted_reference, slot_index) for every slot that has BOTH a
        probation holder to verify and a trusted holder to verify it against. The auditor
        challenges each pair with the same input and compares outputs. A slot with no trusted
        holder yet can't be audited (bootstrap seeds trust); a slot with no probation holder
        needs no audit."""
        with self._lock:
            pairs: List[Tuple[str, str, int]] = []
            for slot in self.topo.slots:
                hs = self.topo.holders(slot.index)          # READY, trusted-first
                trusted = [n for n in hs if n.trust == TRUSTED]
                probation = [n for n in hs if n.trust == PROBATION]
                if not trusted or not probation:
                    continue
                ref = trusted[0].node_id
                for p in probation:
                    pairs.append((p.node_id, ref, slot.index))
            return pairs

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
                                  "trust": self.topo.nodes[h].trust,
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
    def node_endpoints(self) -> List[Tuple[str, Tuple]]:
        """[(node_id, endpoint), …] for public nodes — the RTT prober snapshots this
        under the lock, then probes UNLOCKED (slow network I/O must not block the
        control plane) and records via set_rtt()."""
        with self._lock:
            return [(nid, n.endpoint) for nid, n in self.topo.nodes.items()
                    if n.reachability == "public"]

    def route_snapshot(self) -> List[Node]:
        """Thread-safe snapshot of the pipeline: the PRIMARY healthy holder for each
        slot, in order. The coordinator pins this for a session's lifetime (session
        affinity → warm KV). Raises if any slot is uncovered (the coverage invariant)."""
        with self._lock:
            return [holders[0] for _slot, holders in self.topo.route()]

    # ── replication: load-balanced session routing ────────────────────────────
    def acquire_route(self, session) -> List[Node]:
        """Pick a route for a NEW session, balancing across each slot's replicas: per slot, the
        LEAST-LOADED routable holder (tie-break = holders() order = RTT/freshness, so a slot's
        closest replica wins when load is equal). Concurrent sessions therefore spread across the
        replicas → R replicas of every slot yield ~R parallel pipelines, and aggregate throughput
        scales with contributors instead of funnelling to one primary. Records the session's load
        (release_route frees it). Re-acquiring for the same session (re-prefill/failover) releases
        its prior route first. Raises on a coverage gap.

        Replication=1 (one holder/slot) → identical to route_snapshot (min of one = that holder),
        so this is a safe default; it only changes behavior once slots have replicas."""
        with self._lock:
            self._release_locked(session)
            route = []
            for _slot, holders in self.topo.route():        # raises on an uncovered slot
                i = min(range(len(holders)),
                        key=lambda j: (self._load.get(holders[j].node_id, 0), j))
                route.append(holders[i])
            for n in route:
                self._load[n.node_id] = self._load.get(n.node_id, 0) + 1
            self._session_load[session] = [n.node_id for n in route]
            return route

    def release_route(self, session) -> None:
        """Free a session's load (call when its pin is dropped — end, reset, or failover).
        Idempotent: a session we don't track is a no-op."""
        with self._lock:
            self._release_locked(session)

    def _release_locked(self, session) -> None:
        for nid in self._session_load.pop(session, []):
            c = self._load.get(nid, 0) - 1
            if c > 0:
                self._load[nid] = c
            else:
                self._load.pop(nid, None)

    def load_snapshot(self) -> Dict[str, int]:
        """Active-session count per holder (observability / /health)."""
        with self._lock:
            return dict(self._load)

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

    def payout_eligible(self) -> List[dict]:
        """[{node_id, wallet, trust}] for live (READY) nodes that declared a payout wallet — the set
        the off-chain payout executor distributes revenue across (∝ stake, weighted/gated by the
        executor via StakePoint). Read-only; the executor polls this, then pays from the treasury.

        The coordinator serves the co-located slice [0, CIRCUIT_COORD_LAYERS) but isn't a registered
        holder, so it's included explicitly when it declares its own payout wallet — otherwise the hub
        does real compute for free."""
        with self._lock:
            out = [{"node_id": nid, "wallet": self.wallets[nid], "trust": n.trust}
                   for nid, n in self.topo.nodes.items()
                   if n.state == READY and self.wallets.get(nid)]
            if self.coordinator_wallet:
                out.insert(0, {"node_id": "coordinator", "wallet": self.coordinator_wallet,
                               "trust": TRUSTED})
            return out

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
