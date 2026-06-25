# Cutover Runbook — monolith 72B → floating coordinator

How to migrate the live 72B mesh (`circuitllm.xyz/dllm`) from the single-coordinator monolith to the
floating coordinator **safely**, on a revenue-serving system. Companion to `docs/FLOATING_COORDINATOR.md`.
Principle: **tie the cutover to a real trigger, validate at the real model + scale, ship in stages,
keep the monolith warm for instant rollback.**

## When to cut over (the trigger)

The floating coordinator buys two things; deploy when one is actually needed:
- **No SPOF / fault tolerance** — needs **≥2 head-only orchestrators** (so a dead orchestrator has a
  survivor to re-home onto) + the holders' worker-global KV. A *single* orchestrator is not an
  improvement (it just moves the SPOF and adds a hop).
- **Throughput scaling** — needs **replication ≥ 2** (parallel lanes) + multiple orchestrators; the
  win grows with concurrency (staging: 1.73× at conc 8, 1.5B). Worthwhile once traffic funnels or
  contributor GPUs are joining.

Until one trigger holds, the monolith is fine. Do NOT cut over just because the code is ready.

## Topology change

```
MONOLITH (today)                          FLOATING (target)
─────────────────                         ─────────────────
coordinator pod:                          control plane (CPU; VPS or small pod): registry only
  head + draft + layers [0,16]            orchestrator pods × N (head-only): embed/norm/lm_head + draft
2–3 stage pods: layers [16,80)            holder pods: layers [0,80), replication R (incl. a [0,16) holder)
gateway → :19200 (the one coordinator)    gateway → control plane.acquire_entry → an orchestrator
```

**Artifacts:** none new. Holders bnb-shard-load any range dynamically from the fp16 checkpoint
(`--shard --quant bnb`, `load_model_shard_bnb`), so a holder can take `[0,16)` (the layers the
coordinator used to co-locate). The orchestrator head bundle uses the **head-only shard load**
(`local_layers=None` + `CIRCUIT_SHARD=1` → `load_model_shard_bnb(model, 0, 0, keep_head=True)`,
~6 GB) — built for this cutover; before it, a head-only node would try to load the full 144 GB.

## Stages (each gates the next; do the cheap de-risks first)

### Step 0 — head-only loading  ✅ done
Engine supports a head-only orchestrator at 72B (shard-load embed/norm/lm_head, layers on meta).
Gate: `test_remote_route` byte-identical with `CIRCUIT_TEST_SHARD=1` (the shard head-only path).

### Step 1 — validate the floating stack at 72B (staging, prod untouched)
The non-negotiable gate. Minimal config that proves the unknowns: 1 control plane + holders covering
`[0,80)` (incl. a `[0,16)` holder) + **1** head-only orchestrator.
- **DoD:** the head bundle fits one card (~6 GB); `coverage_ok`; output **byte-identical** to the
  monolith for the same prompts (run the monolith reference on the side); record single-stream tok/s,
  the extra-hop latency, and head VRAM. If byte-identical fails → STOP (a quant/relay mismatch).

### Step 2 — full floating 72B: ≥2 orchestrators + replication ≥ 2
- **DoD:** aggregate throughput scales 1→2 orchestrators on a *scattered* mesh (the real number at
  72B); **orchestrator-kill mid-request continues** (the SPOF-removal proof); re-home lands on the
  same holders (KV-affinity). Reuse `scripts/staging-throughput.py` (point MODEL at the 72B,
  `CIRCUIT_SHARD=1 CIRCUIT_QUANT=bnb`, holders sized to VRAM).

### Step 3 — observability + payout correctness (before any real traffic)
- Control plane `/health` (`coverage_ok`), `/topology`, `/payouts/eligible`; each orchestrator/holder
  `/health`. Alert on coverage loss / orchestrator count < 2.
- **Payout check:** the payout executor merges the engine `/payouts/eligible` + datamesh. Under
  floating, orchestrators register as `orchestrator` nodes and holders as slice nodes — confirm the
  eligible set + wallets are exactly the intended fleet and revenue still splits correctly (the
  topology changed; the payout split must not silently drop a node or double-count).

### Step 4 — shadow
Run the floating stack alongside the monolith; mirror a small fraction of real traffic to it and
**diff outputs** against the monolith. No client sees floating yet. Gate: outputs match, no errors,
latency within budget over a sustained window.

### Step 5 — canary → cutover
The gateway already supports this: set **`CIRCUIT_CONTROL_URL`** on `inference-gateway.js` → it
resolves an orchestrator via `acquire_entry` and proxies there; **unset → it falls back to the fixed
monolith `ENGINE`**. So:
1. Bring up the floating stack to full coverage + ≥2 orchestrators; verify `/health`.
2. Canary: point a *fraction* of gateway traffic at the control plane (a second gateway instance with
   `CIRCUIT_CONTROL_URL` set, behind a weighted nginx split), watch error rate + latency + payouts.
3. Ramp to 100%. Keep the **monolith pods running + warm** the whole time.

### Rollback (instant, at any point)
**Unset `CIRCUIT_CONTROL_URL` on the gateway and restart it** → it proxies to the monolith `ENGINE`
again (the fallback is built in and tested). The monolith pods were never stopped, so this is a
seconds-level revert. Tear down the floating pods after.

## Config reference (72B floating)

```
# control plane (CPU; reachable host, e.g. the VPS behind nginx, or a small pod)
CIRCUIT_ROLE=control CIRCUIT_MESH=1 CIRCUIT_MESH_LAYERS=80 CIRCUIT_MESH_STAGES=<n>
CIRCUIT_MESH_REPLICATION=2 CIRCUIT_MESH_FP=qwen2.5-72b-bnb CIRCUIT_MESH_VERIFY_SIG=1
CIRCUIT_KEY=<cluster key> CIRCUIT_CONTROL_PORT=18932   # NOTE: coordinator_end=0 (no co-located slice)

# head-only orchestrator (GPU; ×N, ≥2)
CIRCUIT_ROLE=orchestrator CIRCUIT_CONTROL_URL=http://<control>:18932 CIRCUIT_MODEL=Qwen/Qwen2.5-72B-Instruct
CIRCUIT_DRAFT=Qwen/Qwen2.5-0.5B-Instruct CIRCUIT_SHARD=1 CIRCUIT_QUANT=bnb CIRCUIT_DEVICE=cuda
CIRCUIT_KEY=<cluster key> CIRCUIT_API_PORT=18931 CIRCUIT_ORCH_ADVERTISE=<public ip>

# holder (GPU; cover [0,80) at replication R; the run-node-72b.sh path, with coordinator_end=0)
CIRCUIT_ROLE via run-node-72b.sh  --shard --quant bnb --capacity-layers <vram-derived> --model-fp qwen2.5-72b-bnb

# gateway cutover / rollback
CIRCUIT_CONTROL_URL=http://<control>:18932   # set = floating ;  unset = monolith (rollback)
```

Deployment gotchas (proven on staging): daemonize with **tmux, not setsid**; a node **co-located**
with the control plane registers via **localhost** (RunPod has no hairpin NAT to a pod's own external
IP); cross-pod nodes advertise their **RunPod-proxy external** host:port.
