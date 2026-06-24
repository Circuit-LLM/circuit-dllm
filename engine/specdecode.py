"""
specdecode.py — greedy speculative decoding over the split pipeline.

Original implementation. A small draft model proposes K tokens; the target (the
split model) verifies all K in ONE pipeline pass; we accept the longest prefix
the target agrees with (greedy), commit one corrected/bonus token, and roll the
rejected tokens' KV back with truncate_to().

Invariant (the correctness contract): greedy speculative output is
**token-identical to plain greedy decode**, for ANY draft. The draft only
affects speed (acceptance rate), never the result — the target's argmax decides
every committed token.

Indexing (per round; `pos` = first unprocessed position, `head` = last committed
but unprocessed token):
  verify batch = [head, d_1..d_K]  at positions pos..pos+K
  logits[0, j] = target's prediction for position pos+j+1  (verifies d_{j+1})
  accept d_1..d_m where m = leading matches; extra = argmax(logits[0, m])
  commit d_1..d_m + extra (m+1 tokens); new head = extra
  keep KV length = pos + m + 1  (head + accepted drafts); rollback to it

`target` protocol:  forward_tokens(ids,start_pos)->logits[1,T,V], kv_len(), rollback(L)
`draft`  protocol:  prefill(prompt), propose(head,K,start_pos)->list[int], rollback(L)
"""

from __future__ import annotations

from typing import Callable, List, Optional

import torch

from engine.kv import StageKV


def _bump(local: dict, stats: Optional[dict], m: int, K: int) -> None:
    """Record one verification round: m draft tokens accepted out of K proposed.
    Updates the per-call `local` and, if given, a cumulative `stats` accumulator.
    If `stats` carries a bounded "window" deque, the round is also appended there so
    a caller can report RECENT acceptance (a lifetime average hides late drift)."""
    for d in (local, stats):
        if d is not None:
            d["rounds"] += 1
            d["accepted"] += m
            d["proposed"] += K
    if stats is not None:
        w = stats.get("window")
        if w is not None:
            w.append((m, K))


@torch.no_grad()
def speculative_greedy(target, draft, prompt_ids: torch.Tensor, n_new: int,
                       K: int = 4, device: str = "cpu",
                       draft_perturb: Optional[Callable[[int, int], int]] = None,
                       stats: Optional[dict] = None) -> List[int]:
    """Run greedy speculative decoding. Returns the generated token ids.

    If `stats` is given (a dict with rounds/accepted/proposed keys), it is
    INCREMENTED in place — the caller can pass a cumulative accumulator to track
    the draft acceptance rate across many calls (an observability signal: a
    silently-degrading draft shows up as acceptance collapsing toward 0)."""
    # prefill the target (its KV -> prompt length) and grab the first token
    logits = target.forward_tokens(prompt_ids, 0)
    draft.prefill(prompt_ids)
    head = int(logits[0, -1].argmax())
    out = [head]
    pos = target.kv_len()
    # EAGLE hook (needs_hidden): give the head the prompt context, then extend it each round with
    # the committed tokens + their target features so the head attends over the WHOLE sequence.
    # GreedyDraft.needs_hidden is False → these are skipped and the standalone path is byte-identical.
    nh = getattr(draft, "needs_hidden", False)
    if nh:
        draft.eagle_prefill(prompt_ids, target.last_hidden())

    local = {"rounds": 0, "accepted": 0, "proposed": 0}
    while len(out) < n_new:
        drafts = draft.propose(head, K, pos, perturb=draft_perturb,
                               target_hidden=(target.last_hidden() if nh else None))
        batch = torch.tensor([[head] + drafts], device=device)
        tlogits = target.forward_tokens(batch, pos)      # [1, K+1, V]

        m = 0
        for j in range(K):
            if int(tlogits[0, j].argmax()) == drafts[j]:
                m += 1
            else:
                break
        extra = int(tlogits[0, m].argmax())              # corrected (m<K) or bonus (m==K)

        out.extend(drafts[:m] + [extra])
        if nh:
            # commit positions pos..pos+m (head + accepted drafts) with their verify-pass features
            draft.eagle_commit([head] + drafts[:m], target.last_hidden()[:, :m + 1, :])
        keep = pos + m + 1
        target.rollback(keep)
        draft.rollback(keep)
        head = extra
        pos = keep

        _bump(local, stats, m, K)

    speculative_greedy.last_stats = local
    return out[:n_new]


@torch.no_grad()
def speculative_greedy_stream(target, draft, prompt_ids: torch.Tensor, n_new: int,
                             eos_ids=(), K: int = 4, device: str = "cpu",
                             draft_perturb: Optional[Callable[[int, int], int]] = None,
                             stats: Optional[dict] = None):
    """Streaming variant of speculative_greedy: a generator that yields each
    committed token id in order.

    The yielded sequence is byte-for-byte the same tokens speculative_greedy
    would return (and therefore identical to plain greedy) up to the stop
    condition. Generation stops at n_new tokens, or as soon as an EOS id is
    committed — EOS itself is not yielded. The draft only affects speed."""
    eos = set(eos_ids)
    import os as _os, sys as _sys, time as _time
    _dbg = _os.environ.get("CIRCUIT_TTFT_DEBUG")
    _t = _time.time() if _dbg else 0
    logits = target.forward_tokens(prompt_ids, 0)
    if _dbg:
        print(f"[ttft] mesh_prefill={_time.time()-_t:.2f}s kv={target.kv_len()}", file=_sys.stderr, flush=True); _t = _time.time()
    draft.prefill(prompt_ids)
    if _dbg:
        print(f"[ttft] draft_prefill={_time.time()-_t:.2f}s", file=_sys.stderr, flush=True)
    head = int(logits[0, -1].argmax())
    if head in eos:
        return
    yield head
    produced = 1
    pos = target.kv_len()
    nh = getattr(draft, "needs_hidden", False)           # EAGLE context feed (see speculative_greedy)
    if nh:
        draft.eagle_prefill(prompt_ids, target.last_hidden())
    while produced < n_new:
        drafts = draft.propose(head, K, pos, perturb=draft_perturb,
                               target_hidden=(target.last_hidden() if nh else None))
        batch = torch.tensor([[head] + drafts], device=device)
        tlogits = target.forward_tokens(batch, pos)      # [1, K+1, V]
        m = 0
        for j in range(K):
            if int(tlogits[0, j].argmax()) == drafts[j]:
                m += 1
            else:
                break
        extra = int(tlogits[0, m].argmax())              # corrected (m<K) or bonus (m==K)
        committed = drafts[:m] + [extra]
        if nh:
            draft.eagle_commit([head] + drafts[:m], target.last_hidden()[:, :m + 1, :])
        keep = pos + m + 1
        target.rollback(keep)
        draft.rollback(keep)
        head = extra                                     # reused as next batch[0], not re-yielded
        pos = keep
        _bump(None, stats, m, K)                         # stream has no per-call last_stats
        for tok in committed:
            if tok in eos or produced >= n_new:
                return
            yield tok
            produced += 1


# --- in-process implementations (Phase 1 correctness bed) ------------------

class SplitTarget:
    """Target = embedding + split stages + norm + lm_head, with per-stage KV."""

    def __init__(self, model, stages, device: str = "cpu"):
        self.embed = model.model.embed_tokens
        self.norm = model.model.norm
        self.lm_head = model.lm_head
        self.stages = stages
        self.config = model.config
        self.device = device
        self.caches = [StageKV(self.config) for _ in stages]
        self._last_hidden = None

    def kv_len(self) -> int:
        return self.caches[0].get_seq_length()

    def last_hidden(self) -> torch.Tensor:
        """The final pre-norm hidden [1,T,D] from the most recent forward_tokens — the feature
        an EAGLE draft conditions on (the target's argmax still decides every committed token)."""
        return self._last_hidden

    @torch.no_grad()
    def forward_tokens(self, token_ids: torch.Tensor, start_pos: int) -> torch.Tensor:
        h = self.embed(token_ids)
        pos = (torch.arange(token_ids.shape[1], device=self.device) + start_pos).unsqueeze(0)
        for st, c in zip(self.stages, self.caches):
            h = st.forward(h, pos, past_key_values=c, use_cache=True)
        self._last_hidden = h
        return self.lm_head(self.norm(h))

    def rollback(self, length: int) -> None:
        for c in self.caches:
            c.truncate_to(length)


class GreedyDraft:
    """Draft = a full model run greedily with its own KV (+ optional perturb)."""

    needs_hidden = False        # a standalone model — never reads the target's hidden. An
                                # EagleDraft sets this True; the loop then feeds it
                                # target.last_hidden() at the EAGLE hook below (docs/EAGLE.md).

    def __init__(self, model, device: str = "cpu"):
        self.model = model
        self.config = model.config
        self.device = device
        self.cache = StageKV(self.config)

    def kv_len(self) -> int:
        return self.cache.get_seq_length()

    @torch.no_grad()
    def prefill(self, prompt_ids: torch.Tensor) -> None:
        self.model(prompt_ids, past_key_values=self.cache, use_cache=True)

    @torch.no_grad()
    def propose(self, head_tok: int, K: int, start_pos: int,
                perturb: Optional[Callable[[int, int], int]] = None,
                target_hidden=None) -> List[int]:
        # target_hidden is the EAGLE seed feature — a standalone GreedyDraft never reads it.
        assert self.kv_len() == start_pos, (self.kv_len(), start_pos)
        props: List[int] = []
        cur = head_tok
        for i in range(K):
            logits = self.model(torch.tensor([[cur]], device=self.device),
                                past_key_values=self.cache, use_cache=True).logits
            nxt = int(logits[0, -1].argmax())
            if perturb is not None:
                nxt = perturb(i, nxt)
            props.append(nxt)
            cur = nxt
        # process the last proposed token so draft KV length == target's (pos+K+1)
        self.model(torch.tensor([[cur]], device=self.device),
                   past_key_values=self.cache, use_cache=True)
        return props

    def rollback(self, length: int) -> None:
        self.cache.truncate_to(length)
