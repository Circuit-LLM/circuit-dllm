# EAGLE drafting — Design

**Status:** design + offline scaffolding (the head-forward + the loop's hidden-feed are the
GPU-validated pieces). **Gated behind `CIRCUIT_DRAFT_KIND=eagle`; default `model` = today's
0.5B GreedyDraft, byte-identical.**
**Lever:** Front 2 / §2.2 of `SPEED_ROADMAP.md` — the biggest "tokens-per-round" lever.

---

## Why EAGLE, and the number that matters

Our live mesh measured draft **acceptance 0.389 → 2.56 tokens/round** with an *untrained*
0.5B model used as a draft. EAGLE trains a tiny (1–2 layer) head that plugs into the target
and reuses its **own hidden features** + its embedding + its LM head, reaching **~0.7–0.85**
acceptance on the Qwen family. In a high-latency mesh the network round-trip is the fixed
cost, so **acceptance per round-trip is the whole game**, and the draft runs on the
coordinator → **zero added hops**. ~0.85 vs 0.39 ≈ **2× the tokens per round-trip → ~half the
round-trips per token** → a near-direct 2× on single-stream latency, stacking on chain relay.

## The split-model subtlety (this is the key design call)

EAGLE needs the target's hidden features. Where those live decides what's tractable here:

- **EAGLE-1 / EAGLE-2 — condition on the target's FINAL hidden state** (the output of the last
  transformer block, before the LM head). **Our coordinator already has this for free:** every
  verification round, the pipeline returns the final hidden to the coordinator, which applies
  norm + lm_head. So EAGLE-1/2 needs **no extra data movement** — the draft head sits on the
  coordinator and reads the hidden that's already coming back. **→ this is our v1.**
  - EAGLE-2 adds dynamic draft *trees* (verify many branches per pass) — composes directly with
    our §2.1 spec-trees work, and is the natural pairing.
- **EAGLE-3 — fuses LOW/MID/HIGH features** from across the target's layers (+6–12 pp
  acceptance). In our split model those three features sit on **different nodes**. To use it,
  each node holding a low/mid/high layer must **attach its feature to the returning activation**
  so the coordinator can fuse them. A clean extension to the chain return path, but real work.
  **→ v2, after EAGLE-1/2 proves the acceptance jump.**

So: **v1 = EAGLE-1/2 (final hidden), which our architecture already serves; v2 = EAGLE-3
feature collection.** v1 alone gets most of the win (0.39 → ~0.75+).

## Interface change (small, backward-compatible, gated)

`specdecode.py`'s draft protocol today is `prefill / propose(head,K,pos) / rollback` — and the
0.5B `GreedyDraft` is a *standalone* model that never sees the target's hidden. EAGLE needs it.
The minimal, zero-risk change:

- Draft gains a flag `needs_hidden` (GreedyDraft `False`, EagleDraft `True`) and `propose` gains
  an optional `target_hidden=None`.
- The target gains `last_hidden()` — the final hidden `[1,T,D]` from its most recent
  `forward_tokens` (the coordinator already computes it; just expose it).
- In `speculative_greedy*`: **only when `draft.needs_hidden`**, pass `target_hidden =
  target.last_hidden()` into `propose`. GreedyDraft is called with an ignored kwarg and
  `last_hidden()` is never invoked → **the live 0.5B path stays byte-identical.**

This loop edit touches the proven speculative path, so it's implemented + validated **on the
GPU** (where `test_specdecode` / `test_specdecode_stream` confirm the GreedyDraft path is
unchanged and EAGLE's output stays token-identical to greedy — EAGLE only affects *speed*).

## Components

- **`engine/eagle.py` `EagleDraft`** — loads the head (1–2 layers), references the target's
  `embed_tokens` + `lm_head`, and implements `prefill / propose(head,K,pos,target_hidden) /
  rollback` to the same contract as `GreedyDraft`. `needs_hidden=True`. The per-step head
  forward (feature → next token + next feature, autoregressed K times) is the model-specific
  bit validated against the real head on a GPU.
- **Coordinator** — `CIRCUIT_DRAFT_KIND` selects the draft: `model` (default) → `GreedyDraft`
  over `CIRCUIT_DRAFT`; `eagle` → `EagleDraft` over `CIRCUIT_EAGLE_HEAD`. The speculative
  output contract (token-identical to greedy) is unchanged — EAGLE is a faster *draft*, the
  target still decides every committed token.

## Head source (researched 2026-06)

Official implementation: **SafeAILab/EAGLE** (EAGLE-1 ICML'24, EAGLE-2 EMNLP'24, EAGLE-3
NeurIPS'25). Pre-trained heads are published per target model (e.g. `yuhuili/EAGLE*`,
AngelSlim) — **a Qwen2.5-72B head exists** (confirm the exact repo/format at integration; the
head is tokenizer-shared with the 72B). If the only published head is EAGLE-3, it still runs in
a **final-hidden-only mode** for v1 (use just its top-feature path) until v2 collects the fused
features; or train an EAGLE-1/2 head (~2–4 h on 4×H100 with the target's activations).

## Build plan

1. **Offline (done):** this doc + `engine/eagle.py` skeleton + the `CIRCUIT_DRAFT_KIND` switch
   (isolated; the proven loop untouched).
2. **GPU session:** download the head; implement `EagleDraft.propose` against its real
   forward; add the `needs_hidden`/`last_hidden()` loop feed; confirm `test_specdecode*`
   (GreedyDraft byte-identical, EAGLE token-identical to greedy); measure acceptance on the 72B
   (target: 0.39 → ~0.75+).
3. **v2:** EAGLE-3 feature collection — nodes attach low/mid/high features to the returning
   activation; coordinator fuses; +6–12 pp acceptance.
