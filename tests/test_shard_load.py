"""
test_shard_load.py — load ONLY a stage's layers into VRAM (true shard loading).

Loads layers [0,32)+head of the 32B-AWQ via device_map and confirms VRAM is
~this stage's share (not the whole model) and the stage forward works. This is
what lets a model too big for one card (72B) fit across two — load-then-prune
can't (the whole model won't fit to be pruned), and AWQ prune doesn't free VRAM.

Run on RunPod GPU (HF_HOME set to the shared cache):
  HF_HOME=/workspace/hf-cache python3 -m tests.test_shard_load
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from engine.model import load_model_shard  # noqa: E402
from engine.stage import stage_for_range  # noqa: E402
from engine.kv import StageKV  # noqa: E402

MODEL = os.environ.get("CIRCUIT_QUANT_MODEL", "Qwen/Qwen2.5-32B-Instruct-AWQ")
START = int(os.environ.get("CIRCUIT_SHARD_START", "0"))
END = int(os.environ.get("CIRCUIT_SHARD_END", "32"))
OTHER = os.environ.get("CIRCUIT_OTHER_DEVICE", "cpu")


def main():
    print(f"shard-load test — {MODEL} layers [{START},{END})+head other={OTHER}")
    model = load_model_shard(MODEL, START, END, keep_head=True,
                             device="cuda:0", other_device=OTHER)
    vram = torch.cuda.memory_allocated() / 1e9
    n = model.config.num_hidden_layers
    print(f"  loaded {END-START}/{n} layers + head -> {vram:.2f} GB VRAM "
          f"(full 32B is ~19 GB)")

    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = tok("The capital of France is", return_tensors="pt").input_ids.to("cuda:0")
    stage = stage_for_range(model, START, END)
    cache = StageKV(model.config)
    with torch.no_grad():
        h = model.model.embed_tokens(ids)
        pos = torch.arange(ids.shape[1], device="cuda:0").unsqueeze(0)
        out = stage.forward(h, pos, past_key_values=cache, use_cache=True)
    ok = tuple(out.shape) == (1, ids.shape[1], model.config.hidden_size)
    print(f"  stage forward: {tuple(out.shape)} ok={ok} kv={cache.get_seq_length()}")
    assert ok, "sharded stage forward broken"
    assert vram < 14, f"VRAM {vram:.2f}GB not reduced — shard load didn't help"
    print("SHARD LOAD OK — only this stage's layers in VRAM, forward works")


if __name__ == "__main__":
    main()
