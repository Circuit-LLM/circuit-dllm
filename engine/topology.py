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


def plan_stages(layer_budget: int, capacities, replication: int = 1) -> int:
    """Choose the FEWEST contiguous stages that the given node capacities can staff —
    the "fewest-fattest" lever (SPEED_ROADMAP §1.2). Each stage boundary is an inter-node
    network hop, and the measured per-token round on a distributed mesh is dominated by
    hop count (the round is compute/hop-overhead-bound, not network-bandwidth-bound), so
    2 fat nodes (1 hop) beats 4 thin nodes (3 hops). Fewer stages ≈ proportionally fewer
    hops ≈ a shorter round.

    `layer_budget` = layers NOT held by the coordinator = `num_layers - coordinator_end`
    (the part split across nodes). `capacities` = each candidate node's `capacity_layers`
    (VRAM-derived max contiguous layers it can hold). `replication` = holders required per
    slot (so k stages need k·replication assignable nodes).

    Returns the smallest stage count k (≥1) such that the balanced-split layout the Topology
    constructor produces — `rem = budget % k` slots of `ceil(budget/k)` and the rest of
    `budget//k` — can be covered: matching the largest slots to the largest-capacity nodes,
    every slot gets `replication` distinct nodes with capacity ≥ that slot's size. As k grows
    the slots shrink, so more nodes qualify; we return the first feasible k. Raises ValueError
    if no k works (the fattest nodes can't even hold the model split as thin as it goes, or
    too few nodes for the requested replication).

    NOTE: assumes capacity-aware slot assignment OR a homogeneous fleet (identical GPUs —
    the common RunPod bring-up). `_pick_slot` currently fills least-committed-first (not
    fat-node→fat-slot), so a heterogeneous fleet can still mis-assign; see _pick_slot TODO."""
    if layer_budget < 1:
        raise ValueError("layer_budget must be >= 1")
    if replication < 1:
        raise ValueError("replication must be >= 1")
    caps = sorted((int(c) for c in capacities), reverse=True)
    for k in range(1, layer_budget + 1):
        per, rem = divmod(layer_budget, k)   # per==0 once k > layer_budget
        if per == 0:
            break                            # can't make k non-empty contiguous slots
        sizes = [per + 1] * rem + [per] * (k - rem)          # balanced (matches constructor)
        demands = sorted(sizes * replication, reverse=True)  # k·replication demands
        # feasible iff sorted demands fit under sorted capacities (largest-to-largest)
        if len(demands) <= len(caps) and all(d <= c for d, c in zip(demands, caps)):
            return k
    raise ValueError(
        f"cannot cover {layer_budget} layers with capacities {caps} at replication "
        f"{replication}: fattest node holds {caps[0] if caps else 0}, need more/bigger nodes")


# ── bandwidth-proportional layer split (SPEED_ROADMAP §1.2 refinement) ────────
# Decode is memory-bandwidth-bound: a stage's per-token time ≈ (its layers × bytes/layer) /
# its GPU memory bandwidth. In a serial pipeline the SLOWEST stage sets the round, so an EQUAL
# split makes a low-bandwidth card the straggler (measured: an L4 stage holding 32/80 layers was
# ~50ms of a ~77ms round). Sizing each stage's slice ∝ its bandwidth equalizes stage times →
# shorter round → faster single-stream, with NO extra hop. Fewest-fattest picks how MANY stages;
# this picks how BIG each one is. Homogeneous fleets get equal sizes (no change).
#
# Best-effort GB/s by GPU type (HBM/GDDR memory bandwidth, the decode-relevant number). Only the
# RATIOS matter for the split; absolute values are approximate/tunable. Unknown → _DEFAULT_BW.
_DEFAULT_BW = 400.0
_GPU_BW = {
    "h100": 3350.0, "h200": 4800.0, "a100 80": 2039.0, "a100": 1555.0,
    "l40s": 864.0, "l40": 864.0, "a40": 696.0, "rtx 6000 ada": 960.0,
    "rtx a6000": 768.0, "a6000": 768.0, "rtx a5000": 768.0, "a5000": 768.0,
    "rtx a4000": 448.0, "a4000": 448.0, "rtx 4090": 1008.0, "rtx 4080": 717.0,
    "rtx 4000 ada": 360.0, "l4": 300.0, "v100": 900.0, "t4": 320.0,
}


def gpu_bandwidth(name: str) -> float:
    """Best-effort memory bandwidth (GB/s) for a GPU name (e.g. 'NVIDIA L40S'). Substring match,
    longest key first so 'a100 80gb' beats 'a100'. Unknown → _DEFAULT_BW. Used as a stage's
    throughput weight when a node doesn't measure/report its own."""
    n = (name or "").lower()
    for key in sorted(_GPU_BW, key=len, reverse=True):
        if key in n:
            return _GPU_BW[key]
    return _DEFAULT_BW


def plan_weighted_split(total: int, weights, min_size: int = 1):
    """Split `total` contiguous layers into len(weights) integer slices ∝ `weights` (each stage's
    throughput / bandwidth), summing EXACTLY to total, each ≥ min_size. Largest-remainder rounding;
    then enforce min_size by moving a layer from the largest slice to any below min. Pure.

    Equal weights → equal split (≡ the balanced constructor). Heterogeneous weights → the faster
    stage gets proportionally more layers so all stages finish a token at ~the same time."""
    w = [float(x) for x in weights]
    n = len(w)
    if n < 1:
        raise ValueError("need at least one weight")
    if min(w) <= 0:
        raise ValueError("weights must be > 0")
    if total < n * min_size:
        raise ValueError(f"total {total} < {n}×min_size {min_size}: can't give each stage min_size")
    W = sum(w)
    ideal = [total * x / W for x in w]
    sizes = [int(f) for f in ideal]                       # floors
    rem = total - sum(sizes)
    order = sorted(range(n), key=lambda i: ideal[i] - sizes[i], reverse=True)
    for i in range(rem):                                  # hand out the remainder by frac part
        sizes[order[i]] += 1
    # enforce min_size: steal from the largest slice until none is below min
    guard = 0
    while min(sizes) < min_size and guard < 10 * n:
        guard += 1
        lo = min(range(n), key=lambda i: sizes[i])
        hi = max(range(n), key=lambda i: sizes[i])
        if sizes[hi] - 1 < min_size:
            break
        sizes[lo] += 1
        sizes[hi] -= 1
    return sizes


def plan_pipeline_layout(num_layers: int, weights, min_size: int = 1):
    """Operator-facing planner: contiguous (start,end) layer ranges for a full pipeline whose
    stage throughput weights are `weights` (stage 0 = the coordinator's co-located slice, then the
    remote stages in order). Sizes ∝ weights (plan_weighted_split). Returns [(0,e0),(e0,e1),…,(…,num_layers)].

    Use it to set the AWQ deploy ranges for a known heterogeneous fleet — e.g. an L40S coordinator
    + an L4 stage over 80 layers → [(0,59),(59,80)] instead of an even (0,40),(40,80), so the L4
    isn't the straggler. Pass GPU names through gpu_bandwidth() to get the weights."""
    sizes = plan_weighted_split(num_layers, weights, min_size=min_size)
    ranges, s = [], 0
    for sz in sizes:
        ranges.append((s, s + sz))
        s += sz
    return ranges


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
                 route_by_latency: bool = False,
                 slot_sizes: Optional[List[int]] = None):
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
        # fixed contiguous slots over [coordinator_end, num_layers). The remainder is spread
        # one-layer-each across the FIRST `rem` slots (balanced split) rather than dumped on
        # the last slot — so the max slot size is ceil(budget/num_stages), not per+rem. That
        # evens VRAM load across nodes AND lets the fewest-fattest planner (plan_stages) staff
        # a given stage count with smaller nodes (a fat trailing slot would otherwise force
        # more stages). Exact-divide layouts are unchanged.
        # slot_sizes (optional): an explicit per-stage layer count for the stage portion, summing
        # to `budget` — used for a BANDWIDTH-PROPORTIONAL split (plan_weighted_split) so a slow
        # card isn't the straggler. None → the balanced equal split below (byte-identical default).
        budget = num_layers - coordinator_end
        self.slots: List[Slot] = []
        s = coordinator_end
        if slot_sizes is not None:
            if len(slot_sizes) != num_stages or sum(slot_sizes) != budget or min(slot_sizes) < 1:
                raise ValueError(
                    f"slot_sizes {slot_sizes} must have {num_stages} entries ≥1 summing to {budget}")
            for i, sz in enumerate(slot_sizes):
                self.slots.append(Slot(i, s, s + sz)); s += sz
        else:
            per, rem = divmod(budget, num_stages)
            for i in range(num_stages):
                e = s + per + (1 if i < rem else 0)
                self.slots.append(Slot(i, s, e))
                s = e

    @classmethod
    def for_fleet(cls, num_layers: int, coordinator_end: int, model_fp: str,
                  capacities, replication: int = 2, weights=None, **kw) -> "Topology":
        """Build a Topology with the FEWEST stages the fleet's `capacities` can staff
        (plan_stages — the fewest-fattest lever, SPEED_ROADMAP §1.2), instead of a static
        num_stages. The control plane calls this once it knows the joined nodes' capacities
        so a 24GB-class fleet runs as 2 fat stages (1 hop), not 4 thin ones (3 hops).
        `replication` is forwarded to BOTH plan_stages (enough nodes to staff k·rep slots)
        and the Topology (per-slot holder target).

        `weights` (optional): per-stage throughput weights (one per stage, len == the chosen k)
        → a BANDWIDTH-PROPORTIONAL slot split (plan_weighted_split) so a slow card isn't the
        straggler. None → equal split. Equal weights ≡ equal split."""
        k = plan_stages(num_layers - coordinator_end, capacities, replication)
        sizes = None
        if weights is not None:
            if len(weights) != k:
                raise ValueError(f"weights needs {k} entries (one per chosen stage), got {len(weights)}")
            sizes = plan_weighted_split(num_layers - coordinator_end, weights)
        return cls(num_layers, coordinator_end, k, model_fp, replication=replication,
                   slot_sizes=sizes, **kw)

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


def _cli_layout(argv):
    """`python3 -m engine.topology layout <num_layers> <gpu0> <gpu1> ...` — print the
    BANDWIDTH-PROPORTIONAL pipeline layout for a known heterogeneous fleet (stage 0 = the
    coordinator). Each GPU name is mapped to a bandwidth weight (gpu_bandwidth) and the layers
    are split ∝ those weights, then printed as ready-to-paste deploy env so the slow card isn't
    the straggler. A numeric arg is taken as an explicit weight (GB/s) instead of a name."""
    if len(argv) < 2:
        print("usage: python3 -m engine.topology layout <num_layers> <gpu0> <gpu1> ...")
        print('  e.g. python3 -m engine.topology layout 80 "NVIDIA L40S" "NVIDIA L4"')
        return 2
    num_layers = int(argv[0])
    specs = argv[1:]
    weights = [float(x) if x.replace(".", "", 1).isdigit() else gpu_bandwidth(x) for x in specs]
    ranges = plan_pipeline_layout(num_layers, weights)
    print(f"# bandwidth-proportional layout for {num_layers} layers across {len(specs)} stages")
    for i, (g, w, (s, e)) in enumerate(zip(specs, weights, ranges)):
        role = "coordinator" if i == 0 else f"stage {i}"
        print(f"#  {role:<12} {g:<28} bw≈{w:>6.0f} GB/s  ->  layers [{s},{e})  ({e-s} layers)")
    c0, c1 = ranges[0]
    print(f"\n# coordinator:  CIRCUIT_COORD_LAYERS={c0}:{c1}")
    for i, (s, e) in enumerate(ranges[1:], 1):
        print(f"# stage {i}:      CIRCUIT_LAYERS={s}:{e}")
    return 0


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2 and sys.argv[1] == "layout":
        sys.exit(_cli_layout(sys.argv[2:]))
    print("commands: layout <num_layers> <gpu0> <gpu1> ...")
    sys.exit(2)
