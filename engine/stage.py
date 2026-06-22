"""
stage.py — a pipeline stage: a contiguous block of transformer decoder layers.

Original implementation. A Stage holds layers [start, end) of a Qwen2 model and
nothing else (no embedding, no lm_head). Its forward replicates exactly what
`Qwen2Model.forward` does per layer — same rotary embeddings, same causal mask,
same KV cache calls — so a model split across N stages is bitwise-identical to
the single-process model. The mask and rotary embeddings are pure functions of
position_ids + config, so each stage recomputes them locally; only the hidden
state has to cross the wire.

Phase 0 uses Hugging Face's DynamicCache for KV (guarantees identical math to the
reference). A custom rollback-capable KV is a Phase 1 optimization validated
against this path.
"""

from __future__ import annotations

from typing import List, Optional

import torch

from transformers.masking_utils import create_causal_mask
try:
    from transformers.masking_utils import create_sliding_window_causal_mask
except Exception:  # older/newer layouts
    create_sliding_window_causal_mask = None


class Stage:
    def __init__(self, layers: List[torch.nn.Module], layer_indices: List[int],
                 rotary_emb: torch.nn.Module, config):
        """
        layers:        the decoder-layer modules for this stage's block
        layer_indices: their *global* indices (for per-layer mask-type lookup)
        rotary_emb:    the model's rotary embedding module (deterministic; shared shape)
        config:        the model config
        """
        assert len(layers) == len(layer_indices)
        self.layers = layers
        self.layer_indices = layer_indices
        self.rotary_emb = rotary_emb
        self.config = config
        layer_types = getattr(config, "layer_types", None)
        self._has_sliding = bool(layer_types) and "sliding_attention" in layer_types

    def _layer_type(self, global_idx: int) -> str:
        lt = getattr(self.config, "layer_types", None)
        return lt[global_idx] if lt else "full_attention"

    def _masks(self, hidden, position_ids, past_key_values, attention_mask=None):
        mask_kwargs = dict(
            config=self.config,
            inputs_embeds=hidden,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
        if self._has_sliding and create_sliding_window_causal_mask is not None:
            mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
        return mapping

    @torch.no_grad()
    def forward(self, hidden: torch.Tensor, position_ids: torch.Tensor,
                past_key_values=None, use_cache: bool = False,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Run this stage's layer block. Returns the updated hidden state.

        attention_mask: optional 2D padding mask [B, kv_len] (1=real, 0=pad) for
        ragged batched decode — create_causal_mask folds it into the causal mask so
        each row attends only to its own real KV. None (default) = pure causal,
        byte-identical to the single-sequence path."""
        position_embeddings = self.rotary_emb(hidden, position_ids)
        masks = self._masks(hidden, position_ids, past_key_values, attention_mask)
        for gidx, layer in zip(self.layer_indices, self.layers):
            hidden = layer(
                hidden,
                attention_mask=masks[self._layer_type(gidx)],
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
        return hidden


def stage_for_range(model, start: int, end: int) -> Stage:
    """Build a single Stage holding global layers [start, end) of `model`."""
    inner = model.model
    idxs = list(range(start, end))
    return Stage([inner.layers[i] for i in idxs], idxs, inner.rotary_emb, model.config)


def split_model(model, boundaries: List[int]) -> List[Stage]:
    """
    Split a loaded Qwen2ForCausalLM into Stages by layer-index boundaries.
    boundaries=[k] -> two stages: [0,k) and [k,N). boundaries=[a,b] -> three, etc.
    The embedding/norm/lm_head stay on the model object (coordinator's job).
    """
    inner = model.model  # Qwen2Model
    layers = inner.layers
    n = model.config.num_hidden_layers
    cuts = [0] + list(boundaries) + [n]
    stages = []
    for s, e in zip(cuts[:-1], cuts[1:]):
        idxs = list(range(s, e))
        stages.append(Stage([layers[i] for i in idxs], idxs, inner.rotary_emb, model.config))
    return stages
