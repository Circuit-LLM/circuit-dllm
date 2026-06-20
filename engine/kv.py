"""
kv.py — per-stage KV cache.

Original implementation. A pipeline stage holds a contiguous block of layers
that does NOT necessarily start at layer 0, but Hugging Face's
DynamicCache.get_seq_length() reads layer slot 0 — which is empty on any stage
that doesn't own layer 0, so it reports length 0 and the causal mask comes out
wrong. StageKV fixes exactly that: it reports the sequence length from the first
layer this stage actually populated. Everything else (storage, update, the HF
attention/mask plumbing) is inherited unchanged, so decode stays bitwise-correct.

It also adds the operations speculative decoding needs: reset() (new sequence)
and truncate_to(length) (roll the cache back after rejected speculative tokens).
"""

from __future__ import annotations

from transformers.cache_utils import DynamicCache


class StageKV(DynamicCache):
    def __init__(self, config):
        super().__init__(config=config)
        self._config = config

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Report length from the first populated layer (any stage, any offset)."""
        for layer in self.layers:
            sl = layer.get_seq_length()
            if sl:
                return sl
        return 0

    def get_mask_sizes(self, query_length: int, layer_idx: int = 0):
        """(kv_length, kv_offset) from the first populated layer.

        create_causal_mask() calls this with layer_idx=0 to size the attention
        mask. HF's default reads slot 0 — empty on a stage that doesn't own
        layer 0, so it reports "no past" and builds a wrong-shaped mask (single-
        token decode survives via SDPA's is-causal skip; multi-token verify does
        not). Delegating to the first populated layer fixes both.
        """
        for layer in self.layers:
            if layer.get_seq_length():
                return layer.get_mask_sizes(query_length)
        return query_length, 0

    def reset(self) -> None:
        """Drop all cached state — start a fresh sequence."""
        super().__init__(config=self._config)

    def truncate_to(self, length: int) -> None:
        """Roll every layer's K/V back to `length` tokens (speculative rollback).

        Accepts M of K speculative tokens -> truncate_to(prefix_len + M).
        Operates on whatever each CacheLayer exposes for its key/value tensors,
        so it stays correct across HF CacheLayer variants.
        """
        for layer in self.layers:
            if layer.get_seq_length() <= length:
                continue
            for attr in ("keys", "values", "key_cache", "value_cache"):
                t = getattr(layer, attr, None)
                if t is not None and hasattr(t, "shape") and t.dim() >= 3:
                    setattr(layer, attr, t[..., :length, :].contiguous())
