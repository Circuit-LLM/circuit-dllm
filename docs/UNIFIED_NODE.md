# Unified Node — Architecture & Build Plan

> **Goal:** one thing a person installs to contribute hardware to Circuit. It detects
> GPU vs CPU, registers with the network using their wallet, runs the right role, and
> earns — managed from a website. One install, one network, one identity.
>
> This supersedes the split where compute (Python engine) and presence/staking
> (Node client) are two disconnected programs. We keep them as two *processes* but
> ship them as one *product*, with the node-client as the front door and the Python
> engine as the compute backend it supervises.
>
> **Authority vs. supervision (don't confuse them):** the **coordinator is the
> network authority** — it admits nodes, assigns layer slots, orchestrates the
> pipeline, and runs attribution. The **node-client is only the *local* supervisor**
> on an operator's machine: it launches and babysits its own Python worker, holds the
> wallet enrollment, and reports telemetry. It *obeys* the coordinator. "Front door"
> = what the user installs/sees; not a network authority.

---

## 1. Target architecture

```
                              circuitllm.xyz  (central dashboard)
                          wallet login → your nodes · stake · earnings
                                          ▲ telemetry / topology / settle
                                          │
        ┌─────────────────────────────────┴───────────────────────────────┐
        │                       COORDINATOR  (you run it)                   │
        │  engine.api: embed · lm_head · draft · OpenAI API · /v1/chat      │
        │  control plane: /register (ed25519 verify + StakePoint gate)      │
        │  topology: assign layer slots by capacity                         │
        │  registry: node→wallet, attribution, settle()                     │
        └───────▲───────────────────────────────▲──────────────────────────┘
                │ encrypted TCP (activations)    │ /register + heartbeat
        ┌───────┴────────┐               ┌───────┴────────────────┐
        │  NODE (GPU)    │               │  NODE (GPU)  …add N     │
        │  ┌───────────┐ │               │  one container:         │
        │  │node-client│ │  supervises   │   node-client (agent)   │
        │  │  (agent)  │─┼──launches──►  │   + engine.stage_worker │
        │  └───────────┘ │               │   own volume: own model │
        │  engine worker │               │   own node-key (hot)    │
        │  layers X:Y    │               │   payout → your wallet  │
        └────────────────┘               └─────────────────────────┘
```

**Three planes, deliberately separate:**
- **Compute** (`engine.coordinator`/`stage`/`wire`/`specdecode`) — the tensor math. *Never touched by product work.*
- **Control** (`engine.control_server`/`registry`/`topology`) — admission, identity, attribution, settle. *Extended, additively.*
- **Product** (node-client supervisor, StakePoint read, dashboard, payout executor) — *all outside the engine.*

---

## 2. What already exists (reuse, don't rebuild)

| Capability | Where | State |
|---|---|---|
| Distributed split inference | `engine/coordinator.py`,`stage.py`,`wire.py` | **live, proven** |
| Predictive drafting | `engine/specdecode.py` | live |
| Self-contained node image (baked code, self-provision model to own volume) | `~/circuit-image` → `ghcr.io/circuit-llm/circuit-dllm:v2-indep` | **built + proven** (this work) |
| Mesh self-join (`/register` → slot by capacity) | `engine/control_server.py`,`topology.py`,`stage_worker.py` | built, CI-tested, **dormant** (`CIRCUIT_MESH` unset) |
| ed25519 identity verify | `engine/control_server.py::make_ed25519_verifier` | built, **not enforced** |
| Node → payout-wallet binding | `stage_worker --payout-wallet`, `registry.wallets` | built |
| Stake-gated admission **hook** | `engine/registry.py` (admission = allowlist/stake check) | scaffolded |
| Attribution (who served which layers → split CIRC) | `engine/registry.py` | built |
| Payout accounting (`settle()` → `(wallet, amount)` batches) | `engine/registry.py` | built |
| StakePoint on-chain stake reader | `circuit-node-client/lib/stakepoint.js::verifyStake` | **working** (pure RPC reads) |
| Stake check endpoint | `circuit-node-client/lib/server.js::/stake/check` | working (gates `/rpc` today) |
| Worker-supervisor bones | `circuit-node-client/lib/llm-worker.js` | exists, **pointed at retired coordinator** |
| Token-2022 CIRC transfer (for payout exec) | `circuit-data-api/lib/escrow.js` | working |

**Staking is StakePoint (3rd-party) — done.** We only ever *read* a wallet's stake; we never build stake/lock/unlock.

---

## 3. Enrollment & security (how a wallet binds to a node)

You don't *claim* a wallet, you *prove* it. The staking wallet **never lands on the
remote box.**

1. On the dashboard, the operator connects their wallet (Phantom) and **stakes CIRC on StakePoint**.
2. Dashboard → "Add a node" → mints a **one-time claim code**, bound to their wallet, short TTL.
3. They launch the node container (e.g. on RunPod) with the claim code as an env var.
4. The node-client generates a throwaway **node-key** (hot key, low value, lives on the pod) and calls `/register` with: node-key pubkey, capacity, model fingerprint, **payout wallet**, claim code, and an ed25519 signature.
5. Coordinator verifies: signature valid (`make_ed25519_verifier`), claim code unused/unexpired/bound to wallet W, and **`verifyStake(W) ≥ min`** (StakePoint read) → admits, assigns a layer slot, records `node→W`.
6. Node runs; earnings accrue to W; the dashboard shows it live.

**Why the attacks fail:** impersonation → signature (can't sign without the key); replay → one-time/expiring code + coordinator nonce; Sybil/flooding → must stake real CIRC to be admitted; naming someone else's wallet → earnings just go *to that wallet's owner*, nothing to steal.

**MVP shortcut:** before claim-codes, admission can be just "node declares payout wallet W; coordinator checks `verifyStake(W) ≥ min`." Safe (no funds at risk); claim-code + signature is the hardening pass.

---

## 4. Build plan (phased; the live engine stays safe at every step)

Each phase is independently valuable, independently testable, and gated so production
never breaks. Difficulty: **L**ow / **M**ed / **H**igh.

### Phase 0 — Self-contained node *(DONE, proven)*
Custom image: deps + engine code baked, model self-provisioned to the node's **own**
volume, cluster key via env. Two independent pods proved it (own disk, own model,
serving). **Remaining cleanup:** cut production over to the independent pair + retire
old shared-volume pods (paused, optional now).

### Phase 1 — Node-client becomes the supervisor *(one entry point)* — **L/M**
- Repoint `lib/llm-worker.js` to launch `engine.stage_worker` (not the retired WS coordinator).
- Node-client detects GPU vs CPU → picks role.
- Fold node-client into the image → **one container, one process the user runs.**
- Still **static** join (no mesh yet) — derisk supervision first.
- *Engine untouched.* **Deliverable:** run one container → it boots the worker → serves.

### Phase 2 — Turn the mesh on *(dynamic join)* — **M (validation)**
- Enable `CIRCUIT_MESH` on the coordinator; enforce `make_ed25519_verifier`.
- Node registers with capacity → coordinator assigns the layer slice.
- *Engine: flip a flag + enforce existing verifier — additive.* **Deliverable:** a new node joins by registering; "add a GPU" needs no hand-wiring.

### Phase 3 — Stake-gated admission + wallet binding — **L/M**
- Coordinator calls `verifyStake(payout_wallet)` (reuse `stakepoint.js`) at `/register`; admit iff ≥ min.
- Add claim-code issuance (dashboard) + binding (control plane).
- *Engine: feed the existing admission hook a yes/no.* **Deliverable:** only staked operators join; rewards bound to their wallet.

### Phase 4 — Payout execution — **L/M**
- Drain `registry.settle()` batches → `circuit-data-api/lib/escrow.js` Token-2022 transfers → pay staked wallets.
- *Engine: read `settle()` output; no inference-path change.* **Deliverable:** contributors get paid CIRC.

### Phase 5 — Central dashboard — **M**
- Wallet login → your nodes (coordinator topology), your stake (`verifyStake`), your earnings (attribution/`settle`).
- *Reads only; no engine change.* **Deliverable:** manage everything from circuitllm.xyz, from anywhere.

### Phase 6 — Resilience *(later)* — **H**
Coordinator redundancy/failover (today's SPOF), CPU-contributes-a-model role, slashing,
multi-model clusters. Deferred by agreement.

---

## 5. Risks / open questions
- **Mesh has never run in production.** Phase 2 is the real validation gate (dynamic topology under live joins/drops).
- **Coordinator is a single point of failure/control** (pool model accepts this until Phase 6).
- **StakePoint layout is reverse-engineered** (`stakepoint.js`, hardcoded 185-byte offsets) — guard + test; re-map if StakePoint changes.
- **Supervisor robustness** — 5–8 min model loads, crash/restart, node churn, slot reassignment.
- **Two runtimes in one container** (Node + Python + CUDA) — heavier image; fine for containers, already the reality.

## 6. Non-negotiable discipline
- **Never put product/money logic in the inference path** (`coordinator.py`/`stage.py`/`wire.py`/`specdecode.py`). It stays in the control plane (`registry`/`control_server`), the node-client, the data-api, and the dashboard.
- Mesh stays **gated** until each phase is proven; static serving (live today) remains byte-identical.
- Engine changes are **additive only** (flip verifier, feed stake yes/no, drain settle).

## 7. Non-goals (for now)
CPU nodes contributing model layers; permissionless multi-coordinator; slashing;
multiple models. All revisited at Phase 6+.
