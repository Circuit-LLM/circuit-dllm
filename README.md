<div align="center">

# circuit-dllm

**The Circuit decentralized LLM engine. A from-scratch Python/PyTorch inference engine that splits one model's transformer layers across GPUs on separate machines over an encrypted network, keeps it fast with predictive drafting, and serves it behind an OpenAI-compatible API with x402 CIRC micropayments.**

[![Python](https://img.shields.io/badge/python-%3E%3D3.10-brightgreen)](https://python.org)
[![Version](https://img.shields.io/badge/version-0.2.0-blue)](https://github.com/Circuit-LLM/circuit-dllm/releases)
[![Network](https://img.shields.io/badge/network-CIRCUIT-gold)](https://circuitllm.xyz)
[![x402](https://img.shields.io/badge/x402-CIRC%20payments-gold)](https://x402.org)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

[Website](https://circuitllm.xyz) · [Chat the DLLM](https://circuitllm.xyz/dllm) · [Inference API](https://inference.circuitllm.xyz) · [Telegram](https://t.me/circuitllm) · [X / Twitter](https://x.com/CircuitLLM)

</div>

---

**[What it does](#what-it-does)** · **[Architecture](#architecture)** · **[Why split a model](#why-split-a-model)** · **[Predictive drafting](#predictive-drafting)** · **[Quick Start](#quick-start)** · **[Config](#configuration)** · **[API](#api)** · **[Deployment](#deployment)** · **[Correctness](#correctness)** · **[Security](#security)** · **[Layout](#repository-layout)**

---

## What it does

- **Splits one model by layer across machines.** A coordinator holds the token embedding and output head; each **stage** runs a contiguous block of the model's transformer layers and relays the hidden state to the next over an encrypted link. No single machine holds the whole model. This is pipeline (layer) parallelism — written from scratch in PyTorch, not a wrapper around an existing runtime.
- **Keeps the network out of the way with predictive drafting.** A small draft model proposes the next few tokens; the full split model verifies them in a single pipeline pass. The output is **identical to ordinary decoding** — the draft only changes how many tokens are confirmed per network round-trip.
- **OpenAI-compatible.** Drop in any OpenAI client: `/v1/chat/completions` (streaming + non-streaming), `/v1/models`, `/v1/workers`, `/health`.
- **Two ways in.** A **free, rate-limited browser demo** at [circuitllm.xyz/dllm](https://circuitllm.xyz/dllm), and an **x402-paid** programmatic endpoint at [inference.circuitllm.xyz](https://inference.circuitllm.xyz) — pay per call in CIRC, no account, no API key.
- **Live today.** A **32-billion-parameter model (Qwen2.5-32B, 4-bit AWQ)** split across **two separate L4 GPUs over WAN**, decoding coherently.

---

## Architecture

```
                 client (OpenAI-compatible request)
                              │
                              ▼
        ┌──────────────── GPU 1 ─────────────────┐
        │  coordinator: embed · norm · lm_head    │
        │  + small draft model (predictive draft) │
        │  STAGE 0:  layers 0 … N/2-1             │
        └───────────────────┬─────────────────────┘
                            │  hidden state, one encrypted hop per pass
                            ▼
        ┌──────────────── GPU 2 ─────────────────┐
        │  STAGE 1:  layers N/2 … N-1             │
        └───────────────────┬─────────────────────┘
                            │  hidden state returns to the coordinator
                            ▼
              coordinator → norm + lm_head + sample → next token
```

A transformer is **sequential**: layer *k* needs layer *k-1*'s output, so a token can't skip ahead, and every machine boundary it crosses is a network round-trip. The split is arranged so the forward pass crosses the network **once per token, not once per layer** — the single thing that makes a cross-machine split usable instead of a toy.

- The **coordinator** (`engine/coordinator.py`) holds the embedding, final norm, and output head, runs the decode loop, and can co-locate stage 0 in-process so a large model loads once per machine.
- Each **stage worker** (`engine/stage_worker.py`) holds a contiguous block of layers and forwards the hidden state.
- The **wire** (`engine/wire.py`) is custom length-prefixed framing, every frame sealed with **ChaCha20-Poly1305** under a shared key.
- The per-stage **KV cache** (`engine/kv.py`) gives each stage correct attention state and the rollback predictive drafting needs.

---

## Why split a model

Splitting one model across machines is **not** a single-request speedup — it's the opposite. Each network hop adds latency, so for a model that *fits on one GPU*, the split is **slower per token** than running it on that one GPU. The reason to split is **capacity**: running a model **too large for any single card** (a 70B+), where spreading it across several GPUs that each hold a slice is the only way to run it at all. The two-GPU 32B running today is the smallest working version of that — proof the split is coherent and correct, and the foundation that scales straight up to bigger models. (For models that *do* fit one card, replicating the whole model per GPU serves more users; splitting is for models that don't fit.)

---

## Predictive drafting

The split leaves exactly one network crossing in the per-token loop. **Predictive drafting** hides it:

1. A small **draft** model (running locally on the coordinator's GPU, no network) proposes the next *K* tokens.
2. The full split model **verifies all K in a single pipeline pass**, producing its own next-token prediction at each position.
3. We keep the longest prefix the full model agrees with, plus one corrected/bonus token, and roll the rest back.

So one network round-trip yields several tokens instead of one. The committed output is **token-for-token identical to plain greedy decoding** — the full model's argmax decides every token; the draft only affects *speed*, never *which* tokens. Proven for any draft (right or wrong) by the test suite. Enable it by setting `CIRCUIT_DRAFT` to a small model that shares the target's tokenizer.

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

A two-machine correctness/throughput run is wired in `scripts/crosspod-test.sh` and `tests/run_coordinator.py`.

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

The reference deployment splits a 32B (4-bit AWQ) across **two L4-class GPUs**:

- **Machine A** runs `deploy/run-api.sh` — coordinator + co-located stage 0 (lower half of layers) + the OpenAI API + the predictive-draft model.
- **Machine B** runs `deploy/run-stage1.sh` — stage 1 (upper half of layers).

Both are managed by a process supervisor (PM2 / systemd) so they survive restarts. `scripts/sync-runpod.sh` syncs the engine to a GPU host for dev/test. Model weights live on a shared volume so they download once.

> Note: a 4-bit 32B *fits* on one L4, so the two-GPU split is a demonstration of correctness, not a per-request speedup. The split earns its keep for models too large for a single card.

---

## Correctness

The engine is validated against a single-process reference:

- `tests/test_correctness.py` — a model split across stages is **bitwise-identical** to the unsplit model.
- `tests/test_specdecode.py` / `tests/test_specdecode_stream.py` — predictive drafting output is **token-identical to greedy** for any draft (accept-all, all-reject, mixed), including over the wire, with clean EOS handling.
- `tests/test_wire.py` — encrypted frame round-trip + tamper detection.
- `tests/test_kv.py` — per-stage KV length and attention-mask correctness.
- `tests/test_twoproc.py` / `tests/test_twoproc_spec.py` — end-to-end across two processes over a socket.

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
│   ├── coordinator.py   embedding + lm_head + sampling + draft + orchestration
│   ├── stage_worker.py  a pipeline stage (holds a contiguous block of layers)
│   ├── stage.py         stage construction / layer slicing
│   ├── api.py           OpenAI-compatible HTTP front end
│   ├── specdecode.py    predictive drafting (draft proposes, split model verifies)
│   ├── kv.py            per-stage KV cache with rollback
│   ├── wire.py          encrypted length-prefixed framing (ChaCha20-Poly1305)
│   ├── tensors.py       tensor ⇄ bytes serialization
│   ├── model.py         model + shard loading
│   └── log.py           structured logging
├── tests/               correctness + wire + KV + two-process suites
├── deploy/              run-api.sh, run-stage1.sh (process-supervised launch)
├── scripts/             sync-runpod.sh, crosspod-test.sh
└── docs/                architecture + design guides
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
