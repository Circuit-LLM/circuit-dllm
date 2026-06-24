"""
eagle.py — EAGLE-1 draft head for the split pipeline (docs/EAGLE.md, SPEED_ROADMAP §2.2).

EAGLE trains a tiny (1-layer) head that plugs into the target and drafts from the target's
OWN final hidden feature + an embedding + the target's LM head — far higher acceptance than a
standalone small model. The head runs on the coordinator, so it adds NO network hop; higher
acceptance = fewer expensive round-trips, which is the whole game in a high-latency mesh.

This is the **v1, real-head implementation** (SafeAILab/yuhuili EAGLE-1 format). The head is a
single decoder layer with the input-layernorm removed (its input is already the fused feature),
plus an `fc` that fuses [token_embedding ; target_feature] (2D → D), a bundled `embed_tokens`,
and `post_attention_layernorm`. It carries NO lm_head — it reuses the TARGET's. The forward is
implemented from scratch (rotary + MHA + SwiGLU MLP) so it is independent of the installed
transformers' layer API.

Draft recurrence (EAGLE-1, greedy), seeded by the target's final hidden `f`:
    cur = head_tok                       # the committed token at position p
    for i in range(K):
        x   = fc([embed(cur) ; f])       # fuse embedding + feature  → [B,D]
        g   = decoder_layer(x_0..x_i)    # next feature  (causal over this round's tokens)
        tok = argmax(target_lm_head(g))  # next token
        cur, f = tok, g                  # autoregress on the predicted feature

v1 attends only WITHIN the current draft round (re-seeded each round from the target feature);
it does not prefill the head's KV over the prompt. That's a conservative lower bound on EAGLE
acceptance — full-context prefill (v2) only raises it. The speculative output stays
token-identical to greedy regardless: the target's argmax decides every committed token, so the
head only affects SPEED (acceptance), never correctness.
"""
from __future__ import annotations

import json
import math
import os
from typing import Callable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── head architecture (Qwen2-72B EAGLE-1 head; dims from its config.json) ─────────────────
class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        x = x.to(torch.float32)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * x.to(dt))


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


class EagleHead(nn.Module):
    """SafeAILab EAGLE-1 head: fc fusion + ONE decoder layer (no input_layernorm) + bundled
    embed_tokens. No lm_head (the target's is reused). Forward is hand-rolled (rotary + MHA +
    SwiGLU) so it does not depend on the transformers layer API."""

    def __init__(self, d_model=8192, n_heads=64, n_kv=64, intermediate=29568,
                 vocab=152064, eps=1e-6, rope_theta=1e6):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv = n_kv
        self.head_dim = d_model // n_heads
        self.rope_theta = rope_theta
        # NOTE: the head's bundled embed_tokens is intentionally NOT allocated — EagleDraft uses
        # the TARGET's embedding (saves ~2.5 GB VRAM; Qwen2.5 embed + Qwen2.5 feature is also more
        # consistent than mixing the head's Qwen2 embed with a Qwen2.5 feature).
        self.fc = nn.Linear(2 * d_model, d_model, bias=True)
        # one decoder layer (flat names match the head's state_dict: layers.0.*)
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(d_model, n_kv * self.head_dim, bias=True)
        self.v_proj = nn.Linear(d_model, n_kv * self.head_dim, bias=True)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)
        self.gate_proj = nn.Linear(d_model, intermediate, bias=False)
        self.up_proj = nn.Linear(d_model, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, d_model, bias=False)
        self.post_attention_layernorm = _RMSNorm(d_model, eps)
        inv = 1.0 / (rope_theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv, persistent=False)

    def _rope(self, position_ids: torch.Tensor, dtype):
        # position_ids: [T] → cos,sin: [1,1,T,head_dim]
        freqs = position_ids.float()[:, None] * self.inv_freq[None, :].to(position_ids.device)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype)[None, None], emb.sin().to(dtype)[None, None]

    def fuse(self, token_embeds: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
        """fc([token_embedding ; feature]) → [B,T,D]. Both inputs [B,T,D] (embedding supplied by
        the caller from the target's embed_tokens)."""
        return self.fc(torch.cat((token_embeds, feature.to(token_embeds.dtype)), dim=-1))

    def layer(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """One EAGLE decoder layer over [B,T,D] — NO input layernorm (x is the fused feature),
        causal self-attention with rotary, then post_attention_layernorm + SwiGLU MLP."""
        B, T, _ = x.shape
        hd, nH = self.head_dim, self.n_heads
        residual = x
        q = self.q_proj(x).view(B, T, nH, hd).transpose(1, 2)       # [B,nH,T,hd]
        k = self.k_proj(x).view(B, T, self.n_kv, hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv, hd).transpose(1, 2)
        cos, sin = self._rope(position_ids, x.dtype)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(hd)          # [B,nH,T,T]
        causal = torch.triu(torch.full((T, T), float("-inf"), device=x.device, dtype=scores.dtype), 1)
        scores = scores + causal
        attn = torch.softmax(scores.float(), dim=-1).to(x.dtype) @ v
        attn = attn.transpose(1, 2).reshape(B, T, self.d_model)
        x = residual + self.o_proj(attn)
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return residual + x


def load_eagle_head(head_path: str, target_model, device: str = "cuda"):
    """Load a SafeAILab/yuhuili EAGLE-1 head. `head_path` may be a directory, a path to the
    pytorch_model.bin, or an HF repo id. Returns an EagleHead on `device` in the target's
    compute dtype (fp16 on GPU) — its feature/embed line up with the target's lm_head."""
    # resolve to a weights file + optional config.json
    cfg_path = None
    if os.path.isdir(head_path):
        bin_path = os.path.join(head_path, "pytorch_model.bin")
        if os.path.exists(os.path.join(head_path, "config.json")):
            cfg_path = os.path.join(head_path, "config.json")
    elif os.path.isfile(head_path):
        bin_path = head_path
        cand = os.path.join(os.path.dirname(head_path), "config.json")
        cfg_path = cand if os.path.exists(cand) else None
    else:  # treat as an HF repo id
        from huggingface_hub import hf_hub_download
        bin_path = hf_hub_download(head_path, "pytorch_model.bin")
        try:
            cfg_path = hf_hub_download(head_path, "config.json")
        except Exception:
            cfg_path = None

    kw = dict(d_model=8192, n_heads=64, n_kv=64, intermediate=29568, vocab=152064,
              eps=1e-6, rope_theta=1e6)
    if cfg_path:
        c = json.load(open(cfg_path))
        kw.update(d_model=c.get("hidden_size", kw["d_model"]),
                  n_heads=c.get("num_attention_heads", kw["n_heads"]),
                  n_kv=c.get("num_key_value_heads", kw["n_kv"]),
                  intermediate=c.get("intermediate_size", kw["intermediate"]),
                  vocab=c.get("vocab_size", kw["vocab"]),
                  eps=c.get("rms_norm_eps", kw["eps"]),
                  rope_theta=c.get("rope_theta", kw["rope_theta"]))

    dtype = torch.float16 if str(device) != "cpu" else torch.float32
    head = EagleHead(**kw)
    sd = torch.load(bin_path, map_location="cpu", weights_only=True)
    # the head stores its single decoder layer under "layers.0.*"; map to our flat names
    remap = {}
    for k, v in sd.items():
        if k == "embed_tokens.weight":
            continue                                       # dropped — the target's embedding is used
        nk = k
        if k.startswith("layers.0."):
            nk = k[len("layers.0."):]                      # self_attn.* / mlp.* / post_attention_layernorm.*
        nk = nk.replace("self_attn.", "")                  # q_proj / k_proj / v_proj / o_proj
        nk = nk.replace("mlp.", "")                        # gate_proj / up_proj / down_proj
        remap[nk] = v
    missing, unexpected = head.load_state_dict(remap, strict=False)
    # inv_freq is a non-persistent buffer (recomputed) → it's the only acceptable "missing".
    miss = [m for m in missing if not m.endswith("inv_freq")]
    if miss or unexpected:
        raise RuntimeError(f"EAGLE head load mismatch: missing={miss} unexpected={unexpected}")
    return head.to(device=device, dtype=dtype).eval()


class EagleDraft:
    """Draft via an EAGLE-1 head conditioned on the target's final hidden state. Same protocol
    as specdecode.GreedyDraft (kv_len / prefill / propose / rollback), plus needs_hidden=True:
    the speculative loop feeds propose() the target feature that produced the head token."""

    needs_hidden = True

    def __init__(self, head: EagleHead, embed, lm_head, norm, device: str = "cuda"):
        self.head = head
        self.embed = embed            # the TARGET's embed_tokens (head's bundled copy is dropped)
        self.lm_head = lm_head        # the TARGET's lm_head (head carries none)
        self.norm = norm              # the TARGET's final RMSNorm: EAGLE's "feature" is the target's
                                      # POST-norm hidden, so we norm last_hidden() before using it
        self.device = device
        self._len = 0
        # running context: committed token ids and their POST-norm features, aligned by position.
        # head input at position i is fuse(embed(tok[i]), feat[i-1]) → the head attends over the
        # WHOLE sequence (full-context EAGLE), not just the current draft round.
        self._toks: List[int] = []
        self._feats: Optional[torch.Tensor] = None   # [1, P, D] post-norm

    def kv_len(self) -> int:
        return self._len

    def _dt(self):
        return next(self.head.parameters()).dtype

    @torch.no_grad()
    def prefill(self, prompt_ids: torch.Tensor) -> None:
        self._len = int(prompt_ids.shape[1])

    @torch.no_grad()
    def eagle_prefill(self, prompt_ids: torch.Tensor, prompt_hidden: torch.Tensor) -> None:
        """Seed the running context with the prompt's tokens + their (PRE-norm) mesh features.
        Called once after the target's prefill (needs_hidden path)."""
        self._toks = prompt_ids[0].tolist()
        self._feats = self.norm(prompt_hidden.to(self.device, self._dt()))   # [1, n, D] post-norm

    @torch.no_grad()
    def eagle_commit(self, tokens: List[int], hidden: torch.Tensor) -> None:
        """Extend the context with newly-committed tokens (positions pos..pos+m) and their
        (PRE-norm) features from the verify pass. `hidden` is [1, len(tokens), D]."""
        self._toks.extend(int(t) for t in tokens)
        f = self.norm(hidden.to(self.device, self._dt()))
        self._feats = f if self._feats is None else torch.cat([self._feats, f], dim=1)

    @torch.no_grad()
    def propose(self, head_tok: int, K: int, start_pos: int,
                perturb: Optional[Callable[[int, int], int]] = None,
                target_hidden: Optional[torch.Tensor] = None) -> List[int]:
        """Propose K tokens. The head attends over the full committed context: input at position i
        is fuse(embed(tok[i]), feat[i-1]); the head is processed at position start_pos, then K
        draft tokens autoregress. Output stays token-identical to greedy (the target verifies)."""
        self._len = start_pos
        dev, dt = self.device, self._dt()
        toks = self._toks + [int(head_tok)]            # tokens at positions 0..start_pos
        feats = self._feats                            # [1, start_pos, D], positions 0..start_pos-1
        if start_pos < 1 or feats is None:             # degenerate (≤1-token prompt): seed-only fallback
            f = (self.norm(target_hidden[:, -1:, :].to(dev, dt)) if target_hidden is not None
                 else feats[:, -1:, :])
            x = self.head.fuse(self.embed(torch.tensor([[int(head_tok)]], device=dev)).to(dt), f)
            base = start_pos
        else:
            # build the head input for positions 1..start_pos: fuse(embed(tok[i]), feat[i-1])
            in_toks = torch.tensor([toks[1:start_pos + 1]], device=dev)        # positions 1..start_pos
            e = self.embed(in_toks).to(dt)                                     # [1, start_pos, D]
            x = self.head.fuse(e, feats[:, :start_pos, :])                     # [1, start_pos, D]
            base = 1
        props: List[int] = []
        for i in range(K):
            T = x.shape[1]
            pos = torch.arange(base, base + T, device=dev)
            g = self.head.layer(x, pos)[:, -1:, :]     # [1,1,D] predicted POST-norm feature
            tok = int(self.lm_head(g)[0, -1].argmax()) # g already post-norm → lm_head directly
            if perturb is not None:
                tok = perturb(i, tok)
            props.append(tok)
            e2 = self.embed(torch.tensor([[tok]], device=dev)).to(dt)
            x = torch.cat([x, self.head.fuse(e2, g)], dim=1)   # extend with the next draft position
        return props

    def rollback(self, length: int) -> None:
        self._len = length
        if length < len(self._toks):
            self._toks = self._toks[:length]
            if self._feats is not None:
                self._feats = self._feats[:, :length, :]
