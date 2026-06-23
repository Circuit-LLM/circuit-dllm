"""
eagle.py — EAGLE draft head for the split pipeline (docs/EAGLE.md, SPEED_ROADMAP §2.2).

EAGLE trains a tiny (1-2 layer) head that plugs into the target and drafts from the target's
OWN final hidden feature + its embedding + its LM head — far higher acceptance than a
standalone small model (~0.75-0.85 vs our measured 0.39 with the untrained 0.5B). The draft
runs on the coordinator, so it adds NO network hop; higher acceptance = fewer expensive
round-trips, which is the whole game in a high-latency mesh.

Split-model note: EAGLE-1/2 condition on the target's FINAL hidden state, which the
coordinator already has every round (the pipeline returns it before norm+lm_head) → tractable
with zero extra data movement. EAGLE-3's low/mid/high fusion needs features from across the
split layers (v2: nodes attach them to the returning activation). This is the v1 drafter.

EagleDraft matches the specdecode draft protocol (prefill / propose / rollback) so it drops
into speculative_greedy* unchanged — EXCEPT `needs_hidden=True` tells the loop to pass
`target.last_hidden()` into propose(). The speculative output stays token-identical to greedy:
EAGLE only affects speed (the target argmax decides every committed token).

OFFLINE SCAFFOLDING: the interface/contract below is FINAL. The two model-specific pieces —
the head's forward and loading its weights — are wired against the real Qwen2.5-72B EAGLE head
on a GPU (raise NotImplementedError until then, so an accidental enable fails loudly, never
silently-wrong). See docs/EAGLE.md build plan.
"""
from __future__ import annotations

from typing import Callable, List, Optional

import torch


class EagleDraft:
    """Draft via an EAGLE head conditioned on the target's final hidden state. Same protocol
    as specdecode.GreedyDraft (kv_len / prefill / propose / rollback), plus needs_hidden."""

    needs_hidden = True          # → the speculative loop feeds propose() target.last_hidden()

    def __init__(self, head, embed, lm_head, device: str = "cpu"):
        # head: the loaded EAGLE head module (1-2 layers @ target hidden_size).
        # embed / lm_head: the TARGET's own embedding + LM head, which EAGLE reuses.
        self.head = head
        self.embed = embed
        self.lm_head = lm_head
        self.device = device
        self._len = 0

    def kv_len(self) -> int:
        return self._len

    @torch.no_grad()
    def prefill(self, prompt_ids: torch.Tensor) -> None:
        # EAGLE-1/2 carries no standalone KV to prefill — the conditioning feature is supplied
        # per-round by the target. We only track length for the loop's rollback bookkeeping;
        # the seed feature arrives with the first propose() as target_hidden.
        self._len = int(prompt_ids.shape[1])

    @torch.no_grad()
    def propose(self, head_tok: int, K: int, start_pos: int,
                perturb: Optional[Callable[[int, int], int]] = None,
                target_hidden: Optional[torch.Tensor] = None) -> List[int]:
        """Propose K tokens by autoregressing the EAGLE head from the target's final hidden.

        Algorithm (EAGLE-1/2, final-hidden):
            f   = target_hidden[:, -1]                 # seed feature (target's last hidden)
            cur = head_tok
            for i in range(K):
                g    = head(f, self.embed(cur))        # head → next feature   [model-specific]
                tok  = int(self.lm_head(g).argmax())   # reuse target LM head
                tok  = perturb(i, tok) if perturb else tok
                props.append(tok); cur, f = tok, g
            return props

        Returns the K proposed token ids. The head(f, emb) forward signature is model-specific
        (SafeAILab/EAGLE format) → wired + validated against the real head on a GPU.
        """
        if target_hidden is None:
            raise ValueError("EagleDraft.propose requires target_hidden — set needs_hidden=True")
        self._len = start_pos
        raise NotImplementedError(
            "EagleDraft.propose: the head forward is wired against the real EAGLE head on GPU "
            "(docs/EAGLE.md step 2). Interface is final; fill in head(f, emb) per the head repo.")

    def rollback(self, length: int) -> None:
        # EAGLE-1/2 keeps no per-token KV to truncate (it re-seeds from the target feature each
        # round) — rollback is length bookkeeping only.
        self._len = length


def load_eagle_head(head_path: str, target_model, device: str = "cpu"):
    """Load an EAGLE head (1-2 layer module, SafeAILab/EAGLE format) from `head_path` and
    return it. The caller binds it to the target's embed + lm_head:

        head  = load_eagle_head(path, target, device)
        draft = EagleDraft(head, target.model.embed_tokens, target.lm_head, device)

    Loading is wired against the real Qwen2.5-72B EAGLE head on a GPU (the exact class/config
    come from the head repo) — see docs/EAGLE.md."""
    raise NotImplementedError(
        "load_eagle_head: wired against the real Qwen2.5-72B EAGLE head on GPU (docs/EAGLE.md). "
        "Confirm the repo/format, load the head module @ the target's hidden_size, return it.")
