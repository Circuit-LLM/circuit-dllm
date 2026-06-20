"""
model.py — load a Qwen2 model + tokenizer.

Phase 0 loads the whole (small) model in every process and each stage uses only
its layer block. Selective shard loading (so a stage on one L4 holds only its
layers of a 32B) is a Phase 3 task — it doesn't affect correctness, only memory.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


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


def _shard_device_map(config, start: int, end: int, keep_head: bool,
                      gpu: str, other: str):
    """device_map placing only owned layers (+ head if keep_head) on the GPU."""
    n = config.num_hidden_layers
    dm = {
        "model.embed_tokens": gpu if keep_head else other,
        "model.norm": gpu if keep_head else other,
        "model.rotary_emb": gpu,            # tiny, no weights; our stage needs it
        "lm_head": gpu if keep_head else other,
    }
    for i in range(n):
        dm[f"model.layers.{i}"] = gpu if (start <= i < end) else other
    return dm


def load_model_shard(model_id: str, start: int, end: int, keep_head: bool = False,
                     device: str = "cuda:0", other_device: str = "cpu"):
    """Load ONLY layers [start, end) (+ head if keep_head) into VRAM; every other
    module goes to `other_device` (cpu, never materialized on the GPU). Use this
    for models too big to load whole on one card — load-then-prune can't help
    because the whole model won't even fit to be pruned. Peak VRAM ~= this
    stage's share. Owned layers keep their global layer_idx (StageKV handles it).
    """
    config = AutoConfig.from_pretrained(model_id)
    dm = _shard_device_map(config, start, end, keep_head, device, other_device)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map=dm, dtype=torch.float16)
    return model.eval()


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
