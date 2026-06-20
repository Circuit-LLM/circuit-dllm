# Architecture

How `circuit-dllm` splits one model across machines and keeps it fast. This is a
deeper companion to the [README](../README.md).

## The problem

A language model generates text one token at a time, and each token must pass
through **every** layer of the model **in order** — layer *k* needs layer *k-1*'s
output. Split those layers across machines and the activations have to cross the
network at each boundary, so the hardware sits idle waiting on the wire. Get that
wrong and a model that should answer in a second takes a minute. The whole design
is organized around one rule: **cross the network as rarely as possible per token.**

## Pipeline (layer) parallelism

The model is cut **by layer** into contiguous blocks — one **stage** per machine:

```
coordinator (GPU 1)        stage worker (GPU 2)
  embed · norm · lm_head      layers N/2 … N-1
  layers 0 … N/2-1
  + predictive draft model
```

- The **coordinator** (`engine/coordinator.py`) owns the token embedding, the final
  norm, and the output head, plus the decode loop and sampling. It can **co-locate
  stage 0 in-process** (`CIRCUIT_LOCAL_LAYERS`), so a large model loads once per
  machine instead of twice.
- Each **stage worker** (`engine/stage_worker.py`) holds a contiguous block of
  layers and does nothing but run them on the hidden state it's handed.

This is **pipeline parallelism**, not tensor parallelism: we split *between* layers,
not *within* them. Tensor parallelism (sharding each layer's matrices and
all-reducing after every layer) needs an NVLink-class interconnect and is a
non-starter over a network — it would all-reduce ~twice per layer per token. Pipeline
parallelism sends only a small activation vector once per stage boundary per token,
which a real network link can carry.

## The per-token decode loop

For each token:

1. The coordinator embeds the current token id.
2. It runs its co-located stage 0 layers locally.
3. It ships the hidden state over the encrypted wire to stage 1, which runs its
   layers and returns the result.
4. The coordinator applies the final norm + output head and **samples** the next
   token.
5. Repeat.

The forward pass crosses the network **once per token, not once per layer** — that
single property is what makes a cross-machine split usable. The loop is greedy and
synchronous (one token at a time) unless predictive drafting is enabled.

## The wire (`engine/wire.py`)

Custom length-prefixed framing over TCP, every frame sealed with **ChaCha20-Poly1305**
AEAD under a pre-shared key (`CIRCUIT_KEY`, hex), with a fresh 96-bit nonce per frame:

```
[ 4B length ][ 12B nonce ][ ciphertext + 16B auth tag ]
```

A passive observer sees only ciphertext; a tampered frame fails authentication and is
rejected. Message types cover activation relays and KV control (reset / truncate).

## Per-stage KV cache (`engine/kv.py`)

Each stage keeps its **own** KV cache for the layers it holds. A subtle correctness
point: an off-the-shelf cache assumes it holds layer 0, so a stage that holds, say,
layers 32–63 has to override how sequence length and attention-mask sizes are
reported — otherwise multi-token verification computes the wrong mask. The cache also
supports **rollback** (`truncate_to`), which predictive drafting relies on.

## Predictive drafting (`engine/specdecode.py`)

Plain greedy decode waits on the network round-trip for every token. Predictive
drafting hides it:

1. A small **draft** model (local to the coordinator's GPU, no network of its own)
   greedily proposes the next *K* tokens.
2. The full split model **verifies all K + 1 positions in a single pipeline pass**,
   producing its own next-token prediction at each.
3. Accept the longest leading run where the full model's argmax matches the draft's
   guess (call it *m*), commit those *m* plus one corrected/bonus token from the full
   model, and **roll the rejected drafts' KV back** to the committed length.

So one network round-trip yields up to *K + 1* committed tokens instead of one.

**Correctness invariant:** the committed sequence is **token-for-token identical to
plain greedy decoding, for any draft.** The full model's argmax decides every
committed token; the draft only affects *how many* are confirmed per round-trip,
never *which*. This holds for a perfect draft, a useless draft, or anything between —
verified by `tests/test_specdecode.py` and `tests/test_specdecode_stream.py`.

The draft must share the target's tokenizer. A 0.5B drafting for a 32B works; a draft
from a different model family would need its own small sibling.

## Performance characteristics

- Latency is dominated by the **network round-trip per (group of) tokens**, not GPU
  compute — so throughput tracks round-trips, and predictive drafting (more tokens
  per round-trip) is the main lever.
- The split is **not** a single-request speedup. For a model that fits one GPU it is
  *slower* per token than that one GPU (you've added a hop). Its value is **capacity**
  — running a model too large for any single card.
- On the reference deployment (a 32B 4-bit AWQ across two L4s over WAN): roughly
  **~10 tok/s** greedy, **~13–14 tok/s** with predictive drafting on, content-dependent
  (the speedup scales with how often the draft guesses right).

## Scaling

- **More machines.** The same layer-cut extends to *N* stages, which is what lets the
  engine serve models far too large for one card — a 70B+ spread across several GPUs
  that each hold a slice.
- **The drafting forest (roadmap).** Checking a *batch* of guesses costs the full
  model almost as little as checking one, so it can verify a whole **tree** of
  candidate continuations per pass. The coordinator keeps its own local draft as a
  **floor**, so speed never depends on the network; additional drafts from remote
  nodes are a best-effort **bonus** — used when they arrive in time, ignored when
  late, so a slow or disconnected drafter can only help, never hurt.
- **Replication vs splitting.** For a model that *fits* one card, running a whole copy
  per GPU (replication) serves more concurrent users; splitting is for models that
  don't fit. They're complementary modes, not competitors.

## Where things live

| Concern | Module |
|---|---|
| Decode loop + orchestration + sampling | `engine/coordinator.py` |
| A pipeline stage (a layer block) | `engine/stage_worker.py`, `engine/stage.py` |
| OpenAI-compatible HTTP API | `engine/api.py` |
| Predictive drafting | `engine/specdecode.py` |
| Per-stage KV cache + rollback | `engine/kv.py` |
| Encrypted wire framing | `engine/wire.py` |
| Tensor ⇄ bytes | `engine/tensors.py` |
| Model + shard loading | `engine/model.py` |
