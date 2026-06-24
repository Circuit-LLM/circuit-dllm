<div align="center">

# circuit-dllm

**The Circuit decentralized LLM engine. A from-scratch Python/PyTorch inference engine that splits one model's transformer layers across GPUs on separate machines over an encrypted network, keeps it fast with predictive drafting, and serves it behind an OpenAI-compatible API with x402 CIRC micropayments.**

[![Python](https://img.shields.io/badge/python-%3E%3D3.10-brightgreen)](https://python.org)
[![Version](https://img.shields.io/badge/version-0.2.0-blue)](https://github.com/Circuit-LLM/circuit-dllm/releases)
[![Network](https://img.shields.io/badge/network-CIRCUIT-gold)](https://circuitllm.xyz)
[![x402](https://img.shields.io/badge/x402-CIRC%20payments-gold)](https://x402.org)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

> **Beta software.** circuit-dllm is under active development. Expect breaking changes between releases, incomplete features, and rough edges. A model split across networked GPUs is only as available as its slowest shard, so validate outputs against the reference model and load-test the full mesh before serving paid traffic.

[Website](https://circuitllm.xyz) · [Chat the DLLM](https://circuitllm.xyz/dllm) · [Inference API](https://inference.circuitllm.xyz) · [Telegram](https://t.me/circuitllm) · [X / Twitter](https://x.com/CircuitLLM)

</div>

---

**[What it does](#what-it-does)** · **[Architecture](#architecture)** · **[Why split a model](#why-split-a-model)** · **[Predictive drafting](#predictive-drafting)** · **[Quick Start](#quick-start)** · **[Config](#configuration)** · **[API](#api)** · **[Deployment](#deployment)** · **[Correctness](#correctness)** · **[Security](#security)** · **[Layout](#repository-layout)**

---

## What it does

- **Splits one model by layer across machines.** A coordinator holds the token embedding and output head; each **stage** runs a contiguous block of the model's transformer layers and relays the hidden state to the next over an encrypted link. No single machine holds the whole model. This is pipeline (layer) parallelism — written from scratch in PyTorch, not a wrapper around an existing runtime.
- **Keeps the network out of the way with predictive drafting.** A small draft model proposes the next few tokens; the full split model verifies them in a single pipeline pass. The output is **identical to ordinary decoding** — the draft only changes how many tokens are confirmed per network round-trip.
- **Open, verified mesh.** Anyone can add a GPU with one command — nodes **register dynamically**, get a layer slice, and earn CIRC. There's no allowlist; instead a new node serves on **probation** and is promoted only after it proves correct compute against a trusted replica (see [Verification](#verification)). Home GPUs behind NAT join through a [relay](docs/RELAY.md), no port-forwarding.
- **OpenAI-compatible.** Drop in any OpenAI client: `/v1/chat/completions` (streaming + non-streaming), `/v1/models`, `/v1/workers`, `/health`.
- **Two ways in.** A **free, rate-limited browser demo** at [circuitllm.xyz/dllm](https://circuitllm.xyz/dllm), and an **x402-paid** programmatic endpoint — pay per call in CIRC, no account, no API key.
- **Live today.** **Qwen2.5-72B-Instruct (4-bit AWQ)** split across **four commodity L4 GPUs over the open internet**, served at [circuitllm.xyz/dllm](https://circuitllm.xyz/dllm). Sub-second time-to-first-token (prefix cache) and ~6–12 tok/s single-stream decode.

---

## Architecture

```
        client (OpenAI request)  ──or──  x402 CIRC gateway
                        │
                        ▼
   ┌──────────── coordinator · GPU 0 ─────────────┐
   │  embed · norm · lm_head · draft · prefix cache │
   │  co-located STAGE 0:  layers 0 … a            │
   └───────────────────┬────────────────────────────┘
        one encrypted hop per token, per stage, on a pinned route
        ▼                    ▼                    ▼
   STAGE 1 · GPU 1      STAGE 2 · GPU 2      STAGE 3 · GPU 3   … (any N nodes)
   layers a … b         layers b … c         layers c … N-1
        └────────────────────┴──── hidden state returns ──┘
                        ▼
        coordinator → norm + lm_head + sample → next token
```

A transformer is **sequential**: layer *k* needs layer *k-1*'s output, so a token can't skip ahead, and every machine boundary it crosses is a network round-trip. The split crosses the network **once per stage per token, not once per layer** — and predictive drafting commits several tokens per crossing. The **live mesh** is the 72B over a coordinator + 3 remote L4 stages; the layout scales to any number of nodes.

- The **coordinator** (`engine/coordinator.py`) holds embedding/norm/lm_head, runs the decode loop, co-locates stage 0, and hosts the draft + the system-prompt prefix cache.
- Each **stage worker** (`engine/stage_worker.py`) holds a contiguous block of layers. Nodes **join dynamically**: a worker registers with the coordinator's control plane (`engine/registry.py` + `topology.py`), is assigned a layer slice sized to its VRAM, downloads only that slice, and serves.
- The **wire** (`engine/wire.py`) is length-prefixed framing sealed with **ChaCha20-Poly1305**. In the dynamic mesh each node gets its **own derived key** at registration (revoking one node never re-keys the rest); the static two-machine path uses one shared key.
- The per-stage **KV cache** (`engine/kv.py`) gives each stage correct attention state plus the rollback predictive drafting needs and the prefix cache's path-truncation.

---

## Why split a model

Splitting one model across machines is **not** a single-request speedup — it's the opposite. Each network hop adds latency, so for a model that *fits on one GPU*, the split is **slower per token** than running it on that one GPU. The reason to split is **capacity**: running a model **too large for any single card** — like the **72B** running today across four 24 GB L4s, none of which could hold it alone. That's the point of the mesh: pool commodity GPUs that are individually too small, and run a frontier-class model on them. (For models that *do* fit one card, replicating the whole model per GPU serves more users; splitting is for models that don't fit.)

---

## Predictive drafting

The split leaves exactly one network crossing in the per-token loop. **Predictive drafting** hides it:

1. A small **draft** model (running locally on the coordinator's GPU, no network) proposes the next *K* tokens.
2. The full split model **verifies all K in a single pipeline pass**, producing its own next-token prediction at each position.
3. We keep the longest prefix the full model agrees with, plus one corrected/bonus token, and roll the rest back.

So one network round-trip yields several tokens instead of one. The committed output is **token-for-token identical to plain greedy decoding** — the full model's argmax decides every token; the draft only affects *speed*, never *which* tokens. Proven for any draft (right or wrong) by the test suite. Enable it by setting `CIRCUIT_DRAFT` to a small model that shares the target's tokenizer.

### System-prompt prefix cache (fast first token)

Predictive drafting speeds up *per-token* decode, but the **first** token still waits on the
prefill — and chat apps resend the same (often large) system prompt on every request, so that
prefill dominates time-to-first-token. The prefix cache removes it: each concurrency slot keeps a
**warm session** whose KV holds the longest token prefix it shares with the slot's previous
request. A new request rolls that KV back to the shared prefix and prefills **only the divergent
suffix** (the user's new message), instead of re-running the whole system prompt through the mesh.

The output is unchanged — it's the same KV the full prefill would have produced — and it falls back
to a fresh prefill whenever a stage holding the warm KV isn't reachable. In practice this drops TTFT
from several seconds to **sub-second** on every request after the first. On by default
(`CIRCUIT_PREFIX_CACHE=1`); minimum shared-prefix length is `CIRCUIT_PREFIX_MIN` (default 16 tokens).

---

## Run a GPU node (join the mesh)

Add your GPU to the live mesh with one command — Linux, or Windows via WSL2:

```bash
curl -fsSL https://circuitllm.xyz/join | bash
```

It installs Docker + the NVIDIA Container Toolkit if missing, pulls the GPU image, asks for a payout
wallet, and runs an auto-restarting container. The node detects your GPU, sizes how many layers it
can hold from VRAM, registers at `https://node.circuitllm.xyz`, downloads only its assigned slice,
and serves. Cloud / public-IP GPUs work out of the box; home desktops behind NAT set
`CIRCUIT_RELAY_URL` and join through the [relay](docs/RELAY.md) — no port-forwarding. You never run
a `docker` command directly.

Step-by-step guides: **[docs/SETUP_DESKTOP.md](docs/SETUP_DESKTOP.md)** (a desktop / home GPU, incl.
Windows WSL2) and **[docs/SETUP_RUNPOD.md](docs/SETUP_RUNPOD.md)** (a RunPod cloud GPU). The image is
built from `docker/` and published to `ghcr.io/circuit-llm/gpu-node`.

## Verification

The mesh is **open**, so identity (ed25519) isn't enough — a node could compute *wrong* and poison
inference. Rather than gate who joins, nodes **prove correctness**: a new node is `probation` and is
never the primary for a token; a background auditor periodically challenges it and a trusted replica
of the same slot with the same input and compares outputs within numerical tolerance (cosine +
relative-L2). Enough passes → promoted to `trusted`; failures → evicted. See
[docs/VERIFICATION.md](docs/VERIFICATION.md). The allowlist still exists but is a dormant
kill-switch (`CIRCUIT_MESH_ALLOWLIST`), off by default.

---

## Quick Start

Requires Python ≥ 3.10 and PyTorch. Correctness is validated on CPU (deterministic); production runs on GPU.

```bash
# 1) install
pip install torch transformers gptqmodel cryptography

# 2) run a stage worker (holds the upper half of the layers) on machine B
python -m engine.stage_worker --model <model_id> --prune --layers 32:64 \
  --port 19210 --key "$(cat cluster_key.txt)"

# 3) run the coordinator + OpenAI API (holds embed/head + lower half) on machine A
CIRCUIT_MODEL=<model_id> CIRCUIT_KEY="$(cat cluster_key.txt)" \
CIRCUIT_STAGES=<machineB-host>:19210 CIRCUIT_LOCAL_LAYERS=0:32 \
CIRCUIT_DEVICE=cuda CIRCUIT_API_PORT=18931 CIRCUIT_DRAFT=<small_draft_model> \
python -m engine.api
```

```bash
# call it like any OpenAI endpoint
curl localhost:18931/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello!"}],"stream":true}'
```

That's the minimal **static** two-machine demo. To run the **dynamic mesh** — how prod runs and how
contributors join — start a coordinator with `deploy/run-coordinator-72b-awq.sh` and add GPU nodes
with the [one-line installer](#run-a-gpu-node-join-the-mesh) or `deploy/run-contributor-node.sh`. A
two-machine correctness/throughput run is wired in `scripts/crosspod-test.sh` and `tests/run_coordinator.py`.

---

## Configuration

The engine is driven by environment variables (see `deploy/run-api.sh`, `deploy/run-stage1.sh`):

| Variable | Where | Purpose |
|---|---|---|
| `CIRCUIT_MODEL` | both | model id (e.g. a 4-bit AWQ checkpoint) |
| `CIRCUIT_KEY` | both | hex of the shared ChaCha20 wire key — must match on every machine |
| `CIRCUIT_LOCAL_LAYERS` | coordinator | layer range to co-locate in-process (e.g. `0:32`) |
| `CIRCUIT_STAGES` | coordinator | comma-separated `host:port` of the remote stage workers |
| `CIRCUIT_DRAFT` | coordinator | small draft model for predictive drafting (omit → plain greedy) |
| `CIRCUIT_DEVICE` | both | `cuda` or `cpu` |
| `CIRCUIT_API_PORT` | coordinator | OpenAI API port (default 18931) |
| `HF_HOME` | both | Hugging Face cache location |
| `CIRCUIT_MESH` | coordinator | `1` = **dynamic mesh** (registry + control plane) instead of the static `CIRCUIT_STAGES` list — how prod runs |
| `CIRCUIT_CONTROL_URL` | node | coordinator control endpoint a node registers with (mesh mode) |
| `CIRCUIT_AWQ_SHARDS` | node | published repo to pull the node's assigned AWQ slice from (AWQ-per-node) |
| `CIRCUIT_RELAY_URL` | node | join via the [NAT relay](docs/RELAY.md) (home GPUs, no port-forward) |
| `CIRCUIT_VERIFY` | coordinator | `1` = run the trustless-[verification](docs/VERIFICATION.md) auditor (default off) |
| `CIRCUIT_PREFIX_CACHE` | coordinator | system-prompt prefix cache (default on) |

The static-path variables above (`CIRCUIT_STAGES`, `CIRCUIT_LOCAL_LAYERS`, shared `CIRCUIT_KEY`) drive the minimal two-machine demo; the dynamic-mesh variables drive the live, contributor-joinable network. Full deploy env lives in `deploy/` (`run-coordinator-72b-awq.sh`, `run-contributor-node.sh`, `run-mesh.sh`).

---

## API

| Method | Path | Notes |
|---|---|---|
| `GET` | `/health` | model id + stage count |
| `GET` | `/v1/models` | served model |
| `GET` | `/v1/workers` | live stage topology (layer ranges per machine) |
| `POST` | `/v1/chat/completions` | OpenAI-compatible, streaming + non-streaming |

Payment is **not** enforced by the engine itself — the engine is internal. Access is gated in front of it: a free, rate-limited browser demo, and an **x402** gateway (`/v1/chat/completions` returns `402` with a CIRC price quote; pay on-chain via a Token-2022 transfer to the treasury; re-send with the transaction signature and the model answers).

---

## Deployment

The **live deployment** splits **Qwen2.5-72B (4-bit AWQ)** across a coordinator + **3 remote L4 stages** (80 layers as `[0,20) [20,40) [40,60) [60,80)`), as a **dynamic mesh**:

- **Coordinator** runs `deploy/run-coordinator-72b-awq.sh` — embed/norm/lm_head + co-located stage 0 + the draft + prefix cache + the control plane (`/register`, `/health`, `/topology`) nodes join through.
- **Each GPU node** runs `deploy/run-contributor-node.sh` (or the one-line installer / Docker image) — registers, is assigned a layer slice sized to its VRAM, pulls only that slice (`CIRCUIT_AWQ_SHARDS`), and serves.

Stages run under a supervisor (systemd / Docker `--restart`) so they survive restarts. A node that dies is reaped and its slot reassigned; replication lets sessions spread across replicas. `scripts/sync-runpod.sh` syncs the engine to a GPU host for dev/test.

> The 72B can't fit any single 24 GB L4, so the split is what makes it runnable at all — that's the point of the mesh (pool small commodity GPUs for a model none of them could hold). A two-machine static split (`deploy/run-api.sh` + `deploy/run-stage1.sh`) is the minimal demo of the same mechanism.

---

## Correctness

The engine is validated against a single-process reference:

- `tests/test_correctness.py` — a model split across stages is **bitwise-identical** to the unsplit model.
- `tests/test_specdecode.py` / `tests/test_specdecode_stream.py` — predictive drafting output is **token-identical to greedy** for any draft (accept-all, all-reject, mixed), including over the wire, with clean EOS handling.
- `tests/test_wire.py` — encrypted frame round-trip + tamper detection.
- `tests/test_kv.py` — per-stage KV length and attention-mask correctness.
- `tests/test_twoproc.py` / `tests/test_twoproc_spec.py` — end-to-end across two processes over a socket.
- `tests/test_topology.py` / `tests/test_registry.py` — slotting, the coverage invariant, failover, churn, admission, per-node keys, attribution.
- `tests/test_trust.py` / `tests/test_verify.py` — trust tiers (probation→trusted, eviction) + the noise-tolerant agreement metric. ([verification](docs/VERIFICATION.md), validated on a live GPU mesh.)
- `tests/test_relay.py` — NAT [relay](docs/RELAY.md) rendezvous: control auth, dial→data pairing, duplex piping.

Pure-logic suites (topology/registry/trust) need no GPU; `test_verify`/`test_relay` need torch/crypto. Run any with `python3 -m tests.<name>`.

---

## Security

- **Encrypted wire.** Every inter-machine frame is sealed with ChaCha20-Poly1305 under a shared key; tampered frames are rejected.
- **Internal engine.** The engine binds to localhost on each machine and is reached only through a controlled gateway, never exposed directly.
- **Paid + free doors.** The public programmatic endpoint is x402-gated (verified on-chain CIRC payment, single-use signatures); the on-site demo is free but origin-locked and per-IP rate-limited.

---

## Repository layout

```
circuit-dllm/
├── engine/
│   ├── coordinator.py   embedding + lm_head + sampling + draft + prefix cache + orchestration
│   ├── stage_worker.py  a pipeline stage (holds a contiguous block of layers) + mesh join client
│   ├── registry.py      control service: admission, per-node keys, attribution, trust
│   ├── topology.py      live layer→node map: slotting, routing, failover, trust tiers
│   ├── verify.py        trustless verification: agreement metric + challenge probe
│   ├── relay.py         NAT relay rendezvous (home GPUs join without port-forwarding)
│   ├── stage.py         stage construction / layer slicing
│   ├── api.py           OpenAI-compatible HTTP front end + control plane + auditor
│   ├── specdecode.py    predictive drafting (draft proposes, split model verifies)
│   ├── kv.py            per-stage KV cache with rollback + tree/prefix path-compaction
│   ├── wire.py          encrypted length-prefixed framing (ChaCha20-Poly1305)
│   ├── tensors.py       tensor ⇄ bytes serialization
│   ├── model.py         model + shard loading
│   └── log.py           structured logging
├── docker/              GPU contributor node image (Dockerfile, entrypoint, build)
├── tests/               correctness, wire, KV, two-process, topology, trust, verify, relay
├── deploy/              run-coordinator-72b-awq.sh, run-contributor-node.sh, run-mesh.sh, …
├── scripts/             sync-runpod.sh, crosspod-test.sh, prov-pod.py, slice-awq.py
└── docs/                ARCHITECTURE, JOIN, RELAY, VERIFICATION, SETUP_DESKTOP, SETUP_RUNPOD, …
```

---

## Part of the Circuit stack

- **[circuit-dllm](https://github.com/Circuit-LLM/circuit-dllm)** — this engine (decentralized LLM inference).
- **[circuit-data-api](https://github.com/Circuit-LLM/circuit-data-api)** — x402-gated Solana data API (and the inference x402 gateway).
- **[circuit-node-client](https://github.com/Circuit-LLM/circuit-node-client)** — the node client that joins the network.
- **[Website](https://circuitllm.xyz)** — circuitllm.xyz, the DLLM chat, and ecosystem visuals.

Predecessor: [`circuit-decentralized-llm-retired`](https://github.com/Circuit-LLM/circuit-decentralized-llm-retired) — the original Node.js/llama.cpp prototype, superseded by this engine.

---

## License

MIT — see [LICENSE](LICENSE).
