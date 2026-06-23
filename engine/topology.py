"""
topology.py — the live source of truth for the pipeline mesh.

Maps the model's transformer layers to the nodes that hold them, with replication,
and answers the two questions the coordinator needs at runtime:

  1. routing — for each pipeline stage, which healthy node serves it (+ fallbacks
     for failover)?
  2. health  — is the model fully covered, and which slots need more replicas?

Pure logic: no torch, no sockets, no I/O. The same code drives the coordinator's
per-token routing and the control service's (re)assignment, and it is unit- and
chaos-testable without GPUs (see tests/test_topology.py).

Model: the layers the coordinator does NOT hold — `[coordinator_end, num_layers)` —
are split into `num_stages` contiguous, fixed SLOTS. Each slot needs `replication`
healthy holders. A node is assigned to exactly one slot. Routing walks the slots in
order and picks a healthy holder per slot; if a holder dies, the next healthy holder
of that slot takes over (failover). The **coverage invariant** — every slot has at
least one healthy holder — is the liveness condition for the whole mesh.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ── node states ──────────────────────────────────────────────────────────────
JOINING = "joining"    # registered + assigned, still downloading its layer weights
READY = "ready"        # weights loaded, serving (the only state we route NEW work to)
SUSPECT = "suspect"    # missed a heartbeat / a request to it failed — transient
DEAD = "dead"          # stably gone (past dead_after_s) — its slot may be reassigned
DRAINING = "draining"  # leaving cleanly: finishes in-flight, takes no new sessions

_ROUTABLE = (READY,)   # only READY nodes receive sessions / count toward coverage

# ── latency estimation (topology-aware routing foundation) ────────────────────
# Neutral RTT (ms) used when neither a measured probe nor a region label is known —
# so an unlabeled node neither wins nor loses proximity routing by default.
_DEFAULT_RTT_MS = 80.0


def _region_rtt_estimate(a: Optional[str], b: Optional[str]) -> float:
    """Coarse RTT (ms) between two region labels like 'na-east' / 'eu-west': exact
    match ≈ local, shared continent (the prefix before '-') ≈ regional, otherwise ≈
    intercontinental. A *bootstrap* heuristic for proximity routing before any probe
    lands — a measured RTT (`Topology.set_rtt`) always supersedes it. Numbers are
    deliberately coarse/tunable; only their ordering matters for routing."""
    if not a or not b:
        return _DEFAULT_RTT_MS
    if a == b:
        return 5.0
    if a.split("-", 1)[0] == b.split("-", 1)[0]:
        return 40.0
    return 150.0


@dataclass
class Node:
    node_id: str                 # ed25519 pubkey (operator identity)
    endpoint: Tuple              # ("host", port) for public, ("relay", relay_id) for NAT
    capacity_layers: int         # how many contiguous layers it can hold (VRAM-derived)
    model_fp: str                # fingerprint of the model the node loaded
    reachability: str = "public" # public | relay
    region: Optional[str] = None # coarse geo label (e.g. "na-east") — a routing HINT,
                                 # superseded by measured RTT; see Topology.rtt()
    payout_wallet: str = ""       # where CIRC earnings settle
    slot: Optional[int] = None    # assigned slot index
    state: str = JOINING
    last_hb: float = 0.0          # last heartbeat timestamp
    wire_key: Optional[bytes] = None  # per-node data-wire key (issued at registration)


@dataclass
class Slot:
    index: int
    start: int                            # layer range [start, end)
    end: int
    holders: List[str] = field(default_factory=list)  # node_ids assigned here


class Topology:
    def __init__(self, num_layers: int, coordinator_end: int, num_stages: int,
                 model_fp: str, replication: int = 2, dead_after_s: float = 30.0,
                 coordinator_region: Optional[str] = None,
                 route_by_latency: bool = False):
        if num_stages < 1:
            raise ValueError("num_stages must be >= 1")
        if not (0 <= coordinator_end < num_layers):
            raise ValueError("coordinator_end must be in [0, num_layers)")
        self.num_layers = num_layers
        self.coordinator_end = coordinator_end
        self.model_fp = model_fp
        self.replication = replication
        self.dead_after_s = dead_after_s
        # topology-aware routing: the coordinator's own region (for region-distance
        # estimates) and whether to order holders by RTT instead of heartbeat freshness.
        # Default off → holder ordering is byte-identical to the pre-latency behavior.
        self.coordinator_region = coordinator_region
        self.route_by_latency = route_by_latency
        self._rtt: Dict[str, float] = {}     # node_id -> measured RTT(ms) from coordinator
        self.nodes: Dict[str, Node] = {}
        # fixed contiguous slots over [coordinator_end, num_layers)
        per = (num_layers - coordinator_end) // num_stages
        self.slots: List[Slot] = []
        s = coordinator_end
        for i in range(num_stages):
            e = num_layers if i == num_stages - 1 else s + per
            self.slots.append(Slot(i, s, e))
            s = e

    def _slot_size(self, slot: Slot) -> int:
        return slot.end - slot.start

    def _committed_count(self, slot: Slot) -> int:
        # holders committed to a slot (not yet DEAD) — a JOINING node counts, since it
        # is provisioning for this slot even before it's READY to route. Counting only
        # READY would pile concurrent joiners into one slot while the rest provision.
        return sum(1 for h in slot.holders
                   if h in self.nodes and self.nodes[h].state != DEAD)

    # ── registration / assignment ────────────────────────────────────────────
    def register(self, node: Node, now: float = 0.0,
                 prefer_range: Optional[Tuple[int, int]] = None) -> Slot:
        """Admit a node and assign it a slot. Normally the slot that needs replicas most and
        that it can hold (_pick_slot). `prefer_range=(start,end)` asks for the slot covering
        exactly those layers — used when a node RE-registers after a coordinator restart so it
        gets back the slot whose weights it ALREADY loaded (else it could be handed a different
        range and serve stale layers). Falls back to _pick_slot if that slot doesn't exist or
        the node can't hold it. Raises on model mismatch / insufficient capacity."""
        if node.model_fp != self.model_fp:
            raise ValueError(f"model mismatch: node {node.model_fp!r} != mesh {self.model_fp!r}")
        # A node re-registering (e.g. its worker restarted — same persisted node_id) must
        # not double-count: drop any prior holding so it isn't listed twice. Removing it
        # before _pick_slot also lets it land back on its old (now-emptier) slot.
        for s in self.slots:
            if node.node_id in s.holders:
                s.holders.remove(node.node_id)
        slot = None
        if prefer_range is not None:
            pr = tuple(prefer_range)
            slot = next((s for s in self.slots if (s.start, s.end) == pr
                         and node.capacity_layers >= self._slot_size(s)), None)
        if slot is None:
            slot = self._pick_slot(node)
        if slot is None:
            raise ValueError("no slot this node can hold (capacity_layers too small)")
        node.slot = slot.index
        node.state = JOINING
        node.last_hb = now
        self.nodes[node.node_id] = node
        slot.holders.append(node.node_id)
        return slot

    def _pick_slot(self, node: Node) -> Optional[Slot]:
        cands = [s for s in self.slots if node.capacity_layers >= self._slot_size(s)]
        if not cands:
            return None
        # fill the least-committed slot first (JOINING counts, so concurrent joiners
        # spread across slots instead of piling into one), tie-break by index
        cands.sort(key=lambda s: (self._committed_count(s), s.index))
        return cands[0]

    # ── health ───────────────────────────────────────────────────────────────
    def heartbeat(self, node_id: str, now: float) -> bool:
        """Refresh a node's liveness. Returns False if the node is UNKNOWN to this topology
        (e.g. the coordinator restarted with an empty registry) — a heartbeating node uses
        that 'False' as the signal to RE-REGISTER (so a coordinator restart doesn't orphan
        already-loaded nodes)."""
        n = self.nodes.get(node_id)
        if n and n.state in (JOINING, READY, SUSPECT, DRAINING):
            n.last_hb = now
            if n.state == SUSPECT:          # a heartbeat clears a transient suspicion
                n.state = READY
            return True
        return False

    def mark_ready(self, node_id: str) -> None:
        """Node finished downloading its layers and is serving."""
        n = self.nodes.get(node_id)
        if n and n.state == JOINING:
            n.state = READY

    def mark_suspect(self, node_id: str) -> None:
        """A request to the node failed, or a beat was missed — don't route to it
        until it recovers (heartbeat) or is reaped (dead)."""
        n = self.nodes.get(node_id)
        if n and n.state == READY:
            n.state = SUSPECT

    def drain(self, node_id: str) -> None:
        n = self.nodes.get(node_id)
        if n:
            n.state = DRAINING

    def reap(self, now: float) -> List[str]:
        """Promote nodes silent past dead_after_s to DEAD (churn damping — only a
        stably-gone node frees its slot). Returns the node_ids newly marked dead."""
        dead = []
        for n in self.nodes.values():
            if n.state in (JOINING, READY, SUSPECT) and now - n.last_hb > self.dead_after_s:
                n.state = DEAD
                dead.append(n.node_id)
        return dead

    def purge(self, now: float) -> List[str]:
        """Drop nodes that have been DEAD long enough to retire from the topology.
        reap() only *marks* DEAD (kept visible for ops for one more dead_after_s
        window); purge then removes them, so holder lists / the node table don't grow
        without bound under long-running churn. Returns the purged node_ids."""
        gone = [nid for nid, n in self.nodes.items()
                if n.state == DEAD and now - n.last_hb > 2 * self.dead_after_s]
        for nid in gone:
            self.remove(nid)
        return gone

    def remove(self, node_id: str) -> None:
        n = self.nodes.pop(node_id, None)
        self._rtt.pop(node_id, None)
        if n and n.slot is not None and node_id in self.slots[n.slot].holders:
            self.slots[n.slot].holders.remove(node_id)

    # ── latency (topology-aware routing foundation) ───────────────────────────
    def set_rtt(self, node_id: str, ms: float) -> None:
        """Record a MEASURED round-trip (ms) from the coordinator to a node (an active
        prober calls this). A measured value supersedes the region estimate in rtt()."""
        self._rtt[node_id] = float(ms)

    def rtt(self, node_id: str) -> float:
        """Estimated RTT (ms) coordinator→node: a measured probe if we have one, else a
        region-distance estimate, else a neutral default. This is the cost a hop to this
        node pays in the star (every hop is coordinator↔holder), so routing minimizes it."""
        if node_id in self._rtt:
            return self._rtt[node_id]
        n = self.nodes.get(node_id)
        return _region_rtt_estimate(self.coordinator_region, n.region if n else None)

    # ── routing ──────────────────────────────────────────────────────────────
    def holders(self, slot_index: int) -> List[Node]:
        """Routable holders for a slot, primary first. Default order is freshest
        heartbeat (liveness). With route_by_latency, the LOWEST-RTT holder is primary —
        i.e. the slot is served by its closest replica to the coordinator (cheapest hop
        in the star), heartbeat freshness breaking ties. The rest are failover fallbacks."""
        slot = self.slots[slot_index]
        live = [self.nodes[h] for h in slot.holders
                if h in self.nodes and self.nodes[h].state in _ROUTABLE]
        if self.route_by_latency:
            live.sort(key=lambda n: (self.rtt(n.node_id), -n.last_hb))
        else:
            live.sort(key=lambda n: -n.last_hb)
        return live

    def route(self) -> List[Tuple[Slot, List[Node]]]:
        """The full pipeline: each slot in order with its ordered routable holders.
        Raises if ANY slot is uncovered — enforces the coverage invariant (a routing
        with a hole would silently corrupt inference, so we refuse it)."""
        plan = []
        for slot in self.slots:
            hs = self.holders(slot.index)
            if not hs:
                raise RuntimeError(
                    f"coverage gap: slot {slot.index} (layers {slot.start}:{slot.end}) "
                    f"has no healthy holder — model not fully covered")
            plan.append((slot, hs))
        return plan

    # ── health summary (for the control service / re-balancer) ────────────────
    def coverage_ok(self) -> bool:
        return all(self.holders(s.index) for s in self.slots)

    def under_replicated(self) -> List[Slot]:
        return [s for s in self.slots if len(self.holders(s.index)) < self.replication]

    def rebalance_targets(self) -> List[Slot]:
        """Slots needing another holder, most-urgent first (uncovered before merely
        under-replicated). The control service assigns new joiners to these."""
        return sorted(self.under_replicated(), key=lambda s: len(self.holders(s.index)))
