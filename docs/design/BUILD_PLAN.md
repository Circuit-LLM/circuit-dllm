# Circuit DLLM — Master Build Plan (2× L4 WAN split → decentralized network)

> The holistic plan. Design doc for the *engine* is
> [TWO_L4_WAN_SPLIT.md](TWO_L4_WAN_SPLIT.md); the *why* is
> [MESH_ARCHITECTURE.md](MESH_ARCHITECTURE.md). This file is about **building it
> end-to-end without forgetting the finishing** — the product, the resilience,
> the operations, the decentralization. Every phase below is "done means done,"
> including its finishing. Nothing ships half-wired.

---

## 0. The end goal (hold this in view the whole time)

Not "two L4s running a 32B." The end goal is:

> **A decentralized network where scattered GPUs each hold part of a model too
> big for any one of them, serve it fast (async-pipelined speculative decoding),
> and anyone can join — accessed through circuitllm.xyz/dllm and an
> OpenAI-compatible API, paid in PISKY, resilient to nodes coming and going.**

The 2× L4 WAN split is the **first concrete instance** of that general network.
So: **design for N scattered stages, instantiate with 2.** Don't hardcode "2."

---

## 1. Definition of Done for the WHOLE system (so nothing is forgotten)

The build is complete only when ALL of these are true — this is the checklist
the "finishing" lives in:

**Inference core**
- [ ] 32B split across 2 stages, output **byte-exact vs a single-GPU reference**.
- [ ] Async-pipelined speculative decoding; measured **~20–30 tok/s** over WAN.
- [ ] Per-stage KV with speculative rollback; correct under CUDA graph capture.

**Product / UX (the part most likely to be forgotten)**
- [ ] circuitllm.xyz/dllm **chats against the new engine** — model shown = 32B,
      topology = 2 nodes, worker strip live, intro modal accurate, mobile usable.
- [ ] **OpenAI-compatible API** works end-to-end with SSE streaming.
- [ ] Errors surface cleanly to the user (no silent hangs, no infinite spinners).

**Resilience (the user's repeated, explicit worry)**
- [ ] A stage dropping **fails the in-flight request cleanly and reconnects with
      bounded backoff — NO crash-loop, no reconnect storm.**
- [ ] Coordinator restart, stage restart, and full reboot all recover cleanly.

**Operations**
- [ ] systemd/PM2 services with correct **bring-up ordering that survives reboot**.
- [ ] Encrypted tunnel/transport managed, auto-reconnecting.
- [ ] **VPS health preserved** — no thread oversubscription, no pegging, swap sane.

**Observability**
- [ ] `/health` + dashboard show: stage liveness, **tok/s, edge RTT**, in-flight
      depth, acceptance rate, session count.

**Decentralization / trust**
- [ ] **Verifiable run receipts** (GPU UUIDs, IPs, regions, RTTs, output hash).
- [ ] **PISKY/x402 payment hooks** wired (live or cleanly stubbed with the seams).
- [ ] A **3rd stage can join** without re-architecting (proves N-stage generality).

**Closure**
- [ ] **Old 7B split-pipeline cluster retired** — we don't run two systems fighting.
- [ ] Operational **runbook** written; **MEMORY.md** updated with the new architecture.

---

## 2. Guiding principles

1. **Finishing is per-phase, not a final phase.** Each phase wires its slice into
   the product and makes it resilient before we call it done.
2. **Correctness before speed.** Byte-exact vs reference *first*; only then optimize.
   A fast wrong model is worthless.
3. **Design N, run 2.** Stage count, layer allocation, and routing are parameters,
   never constants. The 2-L4 build must generalize to scattered N.
4. **Fail loud, recover quiet.** Surface errors to the user/logs; reconnect with
   bounded backoff and jitter — never a tight loop.
5. **Don't break the VPS or the live site while building.** New engine runs
   alongside; cut over only when it's proven.
6. **Two projects stay separate** — this is **circuit**, not pisky. Don't entangle.

---

## 3. Phases (each ends in something complete + finished)

### Phase 0 — Foundations & correctness  *(buildable NOW, single L4)*
Build the skeleton and prove it's correct before any perf or WAN work.
- Python/PyTorch **stage worker**: loads a contiguous layer block, forward pass,
  per-stage **session KV with explicit accept-M/rollback-(K−M)**.
- **Coordinator**: token embedding + lm_head + sampling + a 3B draft (eager first).
- **Encrypted wire**: length-prefixed binary frames, ChaCha20-Poly1305, auth,
  `TCP_NODELAY`, persistent connection.
- Run **both stages on the one L4** (two processes) end-to-end.
- **Finishing for P0:** config system, structured logging, a correctness harness,
  clean process lifecycle (no zombie procs).
- **DoD:** for a stand-in model that fits one L4 (e.g. 7B/14B split 2 ways),
  pipeline output is **byte-exact vs the HF single-process reference**. Greedy
  and sampled paths both verified.

### Phase 1 — Speed: async pipeline + CUDA graphs  *(single L4 / co-located)*
Turn the correct-but-slow pipeline into a fast one.
- **CUDA-graph** the draft (target the ~3.8× shard saw) and the stage forward.
- **Async scheduler**: 2–3 verify chunks in flight; speculative accept; KV
  rollback correct under graph capture (static-address position/KV tensors).
- **Finishing for P1:** acceptance-rate + tok/s instrumentation; a flag to fall
  back to the sync path; assert async output == sync output (still byte-exact).
- **DoD:** async path **token-identical** to sync; measured tok/s on the stand-in
  model; in-flight depth tunable.

### Phase 2 — Product & resilience layer  *(buildable NOW, in parallel with P0/P1)*
The finishing, built early so it's never an afterthought.
- **OpenAI-compatible API** + SSE streaming (adapt the existing `/api/dllm/chat`
  proxy in `circuit/server.js`).
- **Wire circuitllm.xyz/dllm to the new engine**: model badge = 32B, topology =
  2 nodes, worker strip (type/ping/ready), intro modal copy, mobile layout.
  (Reuse the existing site work; just retarget + relabel.)
- **Resilience**: stage-drop → clean request failure + bounded-backoff reconnect
  (no storm); coordinator/stage/reboot recovery; health-gated routing.
- **Observability**: `/health` + dashboard tiles (tok/s, RTT, stage liveness,
  acceptance, sessions).
- **Ops**: systemd/PM2 units, reboot-safe ordering, tunnel management.
- **DoD:** a user chats via site + API against the (co-located) new engine;
  killing a stage degrades gracefully and recovers; dashboard shows live metrics.

### Phase 3 — Go WAN  *(NEEDS the 2nd L4 — provision here)*
The real decentralization test.
- Deploy **stage 1 to the 2nd RunPod** (start same-region, then cross-region).
- Encrypted transport over the open internet; **edge-RTT health monitoring**;
  tune in-flight depth to measured RTT.
- Swap the stand-in model up to the real **Qwen2.5-32B (4-bit), 32/32 split**.
- **Finishing for P3:** WAN reconnect resilience verified by actually killing a
  remote stage mid-stream; dashboard shows cross-node RTT; site still smooth.
- **DoD:** 32B served across **2 WAN L4s at ~20–30 tok/s**, site + API working,
  graceful remote-failure, numbers measured and logged.

### Phase 4 — Decentralize & close out  *(the end goal)*
- **Verifiable receipts** (UUIDs, IPs, regions, RTTs, output hash).
- **PISKY/x402 payment** wired (or seamed for it) with the per-request split.
- **Generalize 2 → N**: dynamic stage join/leave, layer reallocation, prove a
  3rd stage joins live.
- **Retire the old 7B split pipeline**; update runbook + MEMORY.md.
- **DoD:** a 3rd stage joins without code changes; a receipt is produced per run;
  old system decommissioned; docs complete.

---

## 4. What's buildable NOW vs what waits for the 2nd L4

| Now (single L4) | Waits for 2nd L4 |
|---|---|
| P0 foundations + correctness harness | P3 actual WAN deployment |
| P1 async pipeline + CUDA graphs | cross-region RTT tuning |
| P2 entire product/resilience/ops layer | the real 32B at full 2-GPU size |
| Receipts + payment seams (P4 groundwork) | N-stage live join test |

→ We can build and fully validate **P0, P1, P2** (and most of P4's scaffolding)
on the existing single L4 using a stand-in model. The 2nd L4 is needed only to
flip on the real WAN 32B in **Phase 3**. So "once complete we'll get another L4"
fits: I drive P0–P2 to done, then you provision, then we do P3.

---

## 5. Risks & how each is handled

- **Stack change to Python/PyTorch** (from the Node/C runner) — real rebuild of
  the compute path. Mitigation: keep API/site/payment in the existing stack;
  only the stage workers + draft are new Python. Clean seam over the wire.
- **Async correctness is subtle** — mitigation: sync path is the oracle; async
  must match it byte-for-byte before it's trusted.
- **Reconnect storms** (the user's worry) — mitigation: bounded backoff + jitter,
  a circuit-breaker on repeated failures, explicit "no tight loop" tests.
- **Don't destabilize the live VPS/site** — mitigation: new engine runs on a
  separate port/service; cut over only at P3 DoD.
- **Scope creep** — mitigation: the §1 DoD is the contract; nothing outside it
  in this build.

---

## 6. Bottom line

Build order: **P0 correctness → P1 speed → P2 product/resilience** on the one L4
we have; **provision the 2nd L4 → P3 WAN 32B**; then **P4 decentralize + retire
the old system.** Finishing is baked into every phase's DoD, and the §1 master
checklist is the definition of "actually done." The whole thing is designed for
N scattered stages and merely *instantiated* at 2 — so Phase 3 isn't an endpoint,
it's the first node-pair of the real network.
