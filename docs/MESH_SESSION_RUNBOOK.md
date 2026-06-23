# 72B Mesh Session — Runbook

The one paid GPU session that validates Wave A and measures real tok/s. Everything here is
pre-planned so GPU time is *execution*, not figuring-out. Order matters: **do the cheap
de-risks before the 144 GB load.**

Prereqs on the VPS: `RUNPOD_API_KEY` (Infisical), `/tmp/prov72.py` (clean-L4/GPU hunter,
supports `<name> <role> <ports> <min_free> [CLOUD] [DISK] [GPUS] [DC]`), the latest engine
committed. RunPod API needs header `User-Agent: curl/8.5.0`.

---

## 0. Provision (reuse the proven path)

```
# coordinator (api 18931 + control 18932); L4 if available, else RTX 4000 Ada
python3 /tmp/prov72.py circuit-72b-coord coordinator-72b "22/tcp,18931/tcp,18932/tcp" 18000 SECURE 250 \
  "NVIDIA L4,NVIDIA RTX 4000 Ada Generation"
# 3 nodes (stage 19210); SECURE, 180 GB
for n in 1 2 3; do python3 /tmp/prov72.py circuit-72b-node$n node-72b "22/tcp,19210/tcp" 14000 SECURE 180 \
  "NVIDIA L4,NVIDIA RTX 4000 Ada Generation,NVIDIA RTX A4000"; done
```
Note the pod ids, public IPs, and the mapped ports (`19210→…` for nodes, `18931/18932→…` for
coord). Keep the 2 live prod pods (0e78t6cfl72z9y, lt3cweiny0zwwi) untouched.

## 1. Deliver the latest engine to every pod

The baked image (`:v2-indep`) predates chain/EAGLE/P1 — **rsync the current engine** to
`/opt/circuit-engine` on all 4 pods (the dev loop):
```
rsync -rlptz --no-o --no-g -e "ssh -p <ssh> -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no" \
  --exclude .git --exclude __pycache__ --exclude .venv --exclude models \
  ~/circuit-dllm/ root@<ip>:/opt/circuit-engine/
```

## 2. Chain correctness FIRST — cheap, before any 72B load

On ONE pod (has torch), with a 0.5B (~1 GB, ~2 min) — proves chain == star == reference:
```
ssh <coordpod> 'cd /opt/circuit-engine && HF_HOME=/root/hf python3 -m tests.test_chain_relay'
# expect: "CHAIN-RELAY TEST PASSED — chain output byte-identical to star + reference"
```
Also run `python3 -m tests.test_specdecode && python3 -m tests.test_dynamic_relay` (proven
paths still green on real torch). **If chain correctness fails, fix before spending on the 72B.**

## 3. Bring up the 72B mesh (chain + proximity on)

Coordinator (set the new flags; the run script execs `engine.api` which reads them):
```
ssh <coordpod> "
  export CIRCUIT_KEY=$(cat /tmp/circuit_72b_cluster_key.txt)  # or openssl rand -hex 32, save it
  export CIRCUIT_COORDINATOR_ADVERTISE=<coord_public_ip>
  export CIRCUIT_CHAIN=1 CIRCUIT_ROUTE_LATENCY=1 CIRCUIT_REGION=<coord_region>
  export CIRCUIT_ENGINE_DIR=/opt/circuit-engine
  cd /opt/circuit-engine && nohup bash deploy/run-coordinator-72b.sh > /root/coord72.log 2>&1 &"
```
Nodes (each, with its mapped 19210 port + region):
```
ssh <node> "
  export CIRCUIT_CONTROL_URL=http://<coord_public_ip>:<coord_18932_mapped>
  export CIRCUIT_ADVERTISE_HOST=<node_public_ip> CIRCUIT_ADVERTISE_PORT=<node_19210_mapped>
  export CIRCUIT_REGION=<node_region> CIRCUIT_ENGINE_DIR=/opt/circuit-engine
  cd /opt/circuit-engine && nohup bash deploy/run-node-72b.sh > /root/node72.log 2>&1 &"
```
Reminder: in an SSH shell `RUNPOD_PUBLIC_IP`/`RUNPOD_TCP_PORT_19210` are EMPTY — pass
`CIRCUIT_ADVERTISE_HOST/PORT` explicitly. Wait for `/health` `coverage_ok:true` (3 stages).

## 4. Measure — PACKED config sweep (one bring-up, many configs)

The expensive part is the bring-up, so amortize it. Two harness enablers make this cheap:
- **Node re-register on coordinator restart** (heartbeat reports `registered`; a node re-registers
  for its already-loaded slot via `prefer_range`). So changing a *coordinator* config is just
  **restart the coordinator** — the 3 nodes re-join themselves, no node restarts. Kill the old
  coordinator by PID and verify the ports are free first (lingering proc = "Address already in
  use"); relaunch with `setsid bash ... >log 2>&1 </dev/null &`.
- **Per-request `spec_k`** (`SPEC_K=8 scripts/bench-mesh.sh ...`) → sweep K with **no restart**.

The matrix to run from a single mesh (each coordinator-env change = one coordinator restart →
auto re-join → bench):

| Config | env (coordinator) | what it tests |
|---|---|---|
| draft size | `CIRCUIT_DRAFT=Qwen/Qwen2.5-0.5B/1.5B/3B-Instruct` | acceptance (the cheap EAGLE-lite win) |
| chain on/off | `CIRCUIT_CHAIN=0/1` | topology lever (A/B) |
| K sweep | per-request `SPEC_K=4/8/12/16` (no restart) | tokens/round vs round cost |
| continuous batching | `CIRCUIT_BATCH=1` (+`CIRCUIT_MAX_BATCH`) | throughput under load |
| concurrency | `CIRCUIT_MAX_CONCURRENCY=4/8/16` | aggregate scaling |

```
scripts/bench-mesh.sh http://<coord_ip>:<coord_18931_mapped>            # base
SPEC_K=8 scripts/bench-mesh.sh http://<coord_ip>:<coord_18931_mapped>   # K sweep, no restart
CONC="1 8 16" scripts/bench-mesh.sh http://<coord_ip>:<coord_18931_mapped>
```
Record single-stream tok/s, round time, aggregate@concurrency, and `/health` acceptance per
config — against the **1.5 tok/s scattered baseline** (and the measured star 1.77 / chain 1.69).

**Chain-mode caveat (from code review):** chain mode leaks non-head KV (no `CHAIN_KV_CTRL`
yet). The bench is **safe** (~a dozen short gens ≈ tens of MB), but do **not** leave chain mode
serving real traffic until the v1.5 KV-free fix. Also **inject a failover** once: kill the
middle node of a 3-node chain, confirm `/topology` shows the *middle* node SUSPECT (not the
head) and the request recovers — exercises `_suspect_by_endpoint` + re-prefill on the live mesh.

## 5. EAGLE on GPU (the remaining model-specific work)

**Head availability (confirm first):** AngelSlim publishes EAGLE-3 for **Qwen3** + Qwen2.5-VL;
a Qwen2.5-72B-Instruct **text** EAGLE-3 head is NOT confirmed. Options, in order:
  1. find/confirm a published Qwen2.5-72B-Instruct head (HF search `Qwen2.5-72B eagle`),
  2. **train one** (~2-4 h on 4×H100, SafeAILab/EAGLE repo, target activations as supervision),
  3. reconsider base model — **Qwen3 has ready EAGLE-3 heads** (bigger decision; note for operator).

Then:
- `huggingface-cli download <head_repo>` to the coordinator pod.
- Fill in `engine/eagle.py`: `load_eagle_head` (load the head module @ target hidden_size) and
  `EagleDraft.propose` (the `head(f, emb)` forward per the head repo's format).
- Add the loop feed in `engine/specdecode.py`: when `getattr(draft,"needs_hidden",False)`, pass
  `target.last_hidden()` to `propose`; expose `last_hidden()` on `SocketTarget` (store the final
  hidden from the last `forward_tokens`). **GreedyDraft path must stay byte-identical** (guarded
  by `needs_hidden`) — confirm with `test_specdecode` / `test_specdecode_stream`.
- Launch with `CIRCUIT_DRAFT_KIND=eagle CIRCUIT_EAGLE_HEAD=<path>`; confirm speculative output is
  token-identical to greedy; measure acceptance (target **0.39 → ~0.75+**) + tok/s.

## 6. Teardown

```
for pid in <coord> <node1> <node2> <node3>; do
  curl -s -X DELETE -H "Authorization: Bearer $RPK" -H "User-Agent: curl/8.5.0" \
    https://rest.runpod.io/v1/pods/$pid; done
```
Keep prod (32B) running throughout. Record all numbers in `project_circuit_engine` memory.

---

## Expected results (to compare against)

| Config | expectation |
|--------|-------------|
| star, vanilla draft (re-measure baseline) | ~1.5 tok/s cross-DC / ~7 if regionally close |
| + chain relay | ~1.2–1.5× over star (1 round-trip vs N) |
| + EAGLE (acceptance ~0.75+) | ~2× tokens/round on top |
| aggregate @ concurrency 4 | ~3–4× single-stream |

If chain shows < ~1.2× on this fleet, check the node↔node hop latency (the 3 nodes shared one
machine last time → near-zero inter-node; the win then comes from cutting coordinator hops).
