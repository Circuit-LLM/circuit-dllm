# Circuit DLLM — Operating the live engine

Everything we've built is **gated**: the running engine only does what its env flags
turn on, and every capability has a deliberate deploy + rollback. This is the single
place that says what's live, what each flag does, and how to flip one safely.

> Pod path is `/workspace/circuit-engine` (PM2: `circuit-api` on pod1, `circuit-stage1`
> on pod2). Source repo is `~/circuit-dllm` on the VPS (no torch/GPU there). SSH ports
> change on pod restart — read them from `~/runpod.md`.

---

## Current live state (2026-06-22)

| | pod1 `circuit-api` | pod2 `circuit-stage1` |
|---|---|---|
| Role | coordinator + co-located stage `0:32` + API `:18931` + 0.5B draft | remote stage `32:64` |
| Path | static split (`CIRCUIT_STAGES` set, mesh off) | thread-per-peer worker |

**On:** predictive drafting (`CIRCUIT_DRAFT`), pipeline overlap (`CIRCUIT_MAX_CONCURRENCY=4`,
~1.40× under concurrency). **Off (built, gated):** intra-step batching (`CIRCUIT_BATCH`),
the node-join mesh (`CIRCUIT_MESH`).

---

## The flags (all in pod1 `deploy/run-api.sh`; restart to apply)

| Flag | Default | Effect |
|---|---|---|
| `CIRCUIT_MAX_CONCURRENCY` | `1` | >1 = pipeline overlap (per-request connections). **Live: 4.** A lone request is unaffected; concurrency only overlaps when load arrives. |
| `CIRCUIT_BATCH` / `CIRCUIT_MAX_BATCH` | unset / `8` | `=1` routes requests through the batching scheduler (one batched forward per step). **Batched mode is greedy — no draft.** Off today. |
| `CIRCUIT_DRAFT` | unset | a 0.5B draft model → predictive drafting (single-stream latency). **Live: on.** Unused in batch mode. |
| `CIRCUIT_MESH` (+ `_LAYERS/_STAGES/_FP/_REPLICATION/_ALLOWLIST/_SECRET`, `CIRCUIT_CONTROL_PORT`) | unset | hosts the node-join control plane so external GPUs can join (see `JOIN.md §14`). Off today. |

**Invariant:** with a flag at its default the engine behaves byte-identically to before
it existed — each is a separate, regression-tested branch, never a rewrite of the live path.

---

## Deploy procedure (sync new engine code, or flip a flag)

1. **Back up** the engine dir on each pod you touch (rollback point):
   `cp -r /workspace/circuit-engine/engine /workspace/circuit-engine/engine.bak-predeploy`
2. **Sync** only the code: `rsync` local `engine/` → `/workspace/circuit-engine/engine/`
   (leaves `deploy/`, the cluster key, the model cache untouched). Clear `__pycache__`.
   For a flag flip, edit `deploy/run-api.sh` instead.
3. **Restart.** pod2 first if its code changed (`pm2 restart circuit-stage1`), then pod1
   (`pm2 restart circuit-api`). A 32B AWQ reload is **~15–20 min per pod** (parallel if
   both); `/health` only answers once weights are loaded.
4. **Validate at the safe default first** — restart with the new flag *off* (or
   concurrency=1), confirm a test inference matches a known-good baseline, *then* raise
   the flag. (`CIRCUIT_BATCH` / `CIRCUIT_MESH` should be proven on the verify dir's CPU
   tests before a live deploy; throughput is the only thing that needs the GPUs.)

## Rollback (the system must always come back)

- **Bad code:** restore the backup + restart — `rm -rf engine && mv engine.bak-predeploy
  engine && pm2 restart …` on the affected pod(s). Backups currently present on **both pods**.
- **A flag misbehaves:** set it back to its default in `run-api.sh` + `pm2 restart` — no
  code rollback needed (that's the point of gating).

## Isolated verification (never touch the live engine)

CPU tests run in `/workspace/circuit-engine-verify` (a separate dir on pod1). Sync the
repo there and `python3 -m tests.<name>`. The full suite is token-identical / regression
checks; the live dir and process are untouched.

---

## Turning on what's built, when demand arrives

- **More concurrency headroom:** raise `CIRCUIT_MAX_CONCURRENCY` (memory-bounded; watch VRAM).
- **Intra-step batching (the big throughput lever):** prove the verify-dir batch tests,
  then deploy `CIRCUIT_BATCH=1` and **measure batched tok/s on the GPUs** — worth it once
  there's genuine concurrent volume (a swarm on the DLLM, or external traffic). See `BATCHING.md`.
- **Decentralization (outside GPUs join):** set the `CIRCUIT_MESH_*` flags + recruit a
  node (`JOIN.md`). Needs the earning loop (on-chain payout) to give operators a reason.
