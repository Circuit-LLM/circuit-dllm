# Circuit DLLM — trustless node verification

How the mesh stays **open** (anyone runs a GPU, no human approval) while a bad node can't
corrupt inference. The answer is verification, not gating: a new node *proves* it computes
correctly before its output is ever used, and keeps proving it. Design doc for the
implementation in `topology.py` / `registry.py` / `verify.py` + the coordinator auditor.

---

## The problem (and what today's auth does NOT solve)

Registration is ed25519-signed, so a node can't impersonate another's id. That's
**authentication**, not **trustworthiness** — anyone can mint a valid key in a second. An open
mesh therefore has three risks signing doesn't touch:

1. **Bad compute** — a node holds a layer slice and returns wrong/garbage activations, poisoning
   every token routed through it (the pipeline is a series circuit).
2. **Vanishing** — a node grabs an assignment then stalls/dies, breaking the chain.
3. **Visibility** — a node sees the hidden states it processes.

Liveness (#2) is already handled (heartbeat → SUSPECT → DEAD → reassign). Privacy (#3) is
inherent to splitting a model and is out of scope here. **This doc is about #1: correctness.**

## Principle: prove, don't permit

Keep the network open (`allowlist=None`, the default). Demote the allowlist to a dormant
**kill-switch** (set `CIRCUIT_MESH_ALLOWLIST` only to freeze the mesh in an incident). Replace the gate
with two mechanisms that compose with the existing replication:

### 1. Trust tiers (routing safety)

Trust is orthogonal to liveness. A `Node` gains a `trust` tier:

- **`probation`** — every newly-joined node. It is assigned a slot and serves as a **redundant /
  shadow** holder, but is **never the PRIMARY** whose output is used for a response.
- **`trusted`** — promoted after it passes enough verification challenges. Eligible to be primary.

`holders()` orders **trusted-first**, so `route()` picks a trusted primary whenever one exists,
and a probation node only becomes primary as a last resort to preserve coverage (better a
checked-but-unproven node than a coverage hole — and that path is logged). The **seed fleet**
(the operator's bootstrap nodes, `CIRCUIT_MESH_SEED_NODES`) starts `trusted` so there's a reference to
verify against from t=0.

### 2. Challenge verification (earning trust)

A background **auditor** (off the hot path, rate-limited) repeatedly:

1. picks a `probation` node and a `trusted` reference holder of the **same slot**;
2. sends both the **same challenge activation** (a deterministic pseudo-random hidden state at the
   slot's input width) over the normal ACTIVATION wire, on a throwaway session;
3. compares the two outputs with a **noise-tolerant metric** (cosine similarity ≥ `cos_thresh`
   **and** relative-L2 ≤ `l2_thresh`) — honest replicas agree to high cosine even across different
   GPUs/rounding (the mesh is numerically non-deterministic — exact-equality would false-positive),
   while garbage, zeros, wrong-shape, or a materially-wrong computation falls far below;
4. records the result: **pass** → `checks_passed++` (→ `trusted` after `promote_after`); **fail** →
   `strikes++` (→ evicted/`DEAD` after `strike_max`, its slot reassigned).

No reference replica for a slot yet (only probation holders)? That slot can't be audited until a
trusted holder exists — bootstrap covers this (seed = trusted), and a probation node there is
flagged, not silently trusted.

### Why this is enough (and its limits)

- **Catches the real failure mode** — broken, lazy, misconfigured, or blatantly malicious nodes
  return outputs that don't match a trusted replica. They never reach primary, and they're evicted.
- **Trustless** — verification is by agreement with a replica + challenge, not by a human vouching.
- **Cheap** — out-of-band, sampled; zero added latency on real requests.
- **Honest limit:** a *subtle* adversary who introduces errors within the numerical-noise floor can
  evade a single pairwise check. Defenses layer on later: **N-of-M replica voting**, **random
  audit timing**, and **CIRC stake slashing** (economic skin-in-the-game) — see §Future. The tier
  system is the foundation those build on.

## Economic layer (later): stake, not gate

CIRC **stake** gives sybil resistance without an allowlist: a node bonds stake to join the trusted
tier faster, and a verification failure **slashes** it. Reputation (pass rate, uptime, stake)
becomes the routing weight. This is the permissionless end-state; the trust tiers + challenges
above are what it plugs into.

## Components

| Piece | File | Tested |
|------|------|--------|
| `Node.trust`, `checks_passed/strikes`, `record_check`, trusted-first `holders()` | `topology.py` | CPU unit |
| seed-trusted admission, `record_check`, `audit_pairs()`, trust in snapshot | `registry.py` | CPU unit |
| noise-tolerant `outputs_agree()` + challenge tensor | `verify.py` | CPU unit |
| background auditor: challenge a pair, compare, record | coordinator/control | needs mesh |
| allowlist = dormant kill-switch (already default-off) | `registry.py` | — |

**Rollout:** trust tiers + routing ship first (pure, safe — unverified nodes simply never go
primary). The auditor ships **guarded off** (`CIRCUIT_VERIFY=0` default) until validated on a real
multi-node mesh, then enabled. Default behavior with no seed config + verify off is byte-identical
to today.

## Operator knobs

| Env | Default | Effect |
|-----|---------|--------|
| `CIRCUIT_MESH_SEED_NODES` | — | comma-sep node_ids admitted TRUSTED (bootstrap reference) |
| `CIRCUIT_VERIFY` | `0` | `1` starts the background auditor |
| `CIRCUIT_VERIFY_INTERVAL` | `30` | seconds between audit rounds |
| `CIRCUIT_VERIFY_COS` | `0.99` | min cosine similarity to pass a check |
| `CIRCUIT_VERIFY_L2` | `0.05` | max relative-L2 to pass a check |
| `CIRCUIT_MESH_ALLOWLIST` | — | **kill-switch**: set to freeze the mesh to listed node_ids (else open) |

Promotion (`promote_after`, default 3 passes) and eviction (`strike_max`, default 2 strikes) are
`record_check` parameters. Tests: `tests/test_trust.py` (control-plane, CPU) and
`tests/test_verify.py` (metric, needs torch).
