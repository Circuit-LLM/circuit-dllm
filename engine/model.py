"""
model.py — load a Qwen2 model + tokenizer.

Phase 0 loads the whole (small) model in every process and each stage uses only
its layer block. Selective shard loading (so a stage on one L4 holds only its
layers of a 32B) is a Phase 3 task — it doesn't affect correctness, only memory.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_id: str, dtype=None, device: str = "cpu"):
    # GPU: fp16 + device_map (required for quantized AWQ/GPTQ — can't .to() after
    # load). CPU: fp32 + .to(). dtype defaults by device unless overridden.
    if dtype is None:
        dtype = torch.float16 if device != "cpu" else torch.float32
    if device == "cpu":
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map=device)
    return model.eval()


def load_tokenizer(model_id: str):
    return AutoTokenizer.from_pretrained(model_id)


def prune_to_layers(model, start: int, end: int, keep_head: bool = True):
    """Free the VRAM of layers/head this stage doesn't own.

    A stage holding layers [start, end) doesn't need the others, and a
    non-coordinator stage needs neither embedding nor lm_head. We replace the
    unneeded modules with Identity and empty the CUDA allocator so their weights
    are freed: peak memory during load is the whole (4-bit) model, steady-state
    is ~this stage's share — letting two L4s comfortably hold a 32B with KV room.

    The kept layers retain their *global* layer_idx (StageKV handles the offset),
    so stage_for_range(model, start, end) works unchanged afterward.
    """
    import gc
    inner = model.model
    for i in range(len(inner.layers)):
        if not (start <= i < end):
            inner.layers[i] = torch.nn.Identity()
    if not keep_head:
        inner.embed_tokens = torch.nn.Identity()
        inner.norm = torch.nn.Identity()
        model.lm_head = torch.nn.Identity()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model
