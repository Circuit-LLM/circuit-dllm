"""
model.py — load a Qwen2 model + tokenizer.

Phase 0 loads the whole (small) model in every process and each stage uses only
its layer block. Selective shard loading (so a stage on one L4 holds only its
layers of a 32B) is a Phase 3 task — it doesn't affect correctness, only memory.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_id: str, dtype=torch.float32, device: str = "cpu"):
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    model.eval().to(device)
    return model


def load_tokenizer(model_id: str):
    return AutoTokenizer.from_pretrained(model_id)
