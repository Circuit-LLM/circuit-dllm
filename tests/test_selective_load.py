"""
test_selective_load.py — a stage holds only its layers, freeing the rest's VRAM.

Loads the 3B 4-bit, prunes to the second half of layers (+ drops embed/norm/
lm_head, which a non-coordinator stage doesn't need), and confirms (a) VRAM drops
to roughly that stage's share and (b) the pruned stage still runs a forward.
This is how a 32B fits two L4s with KV headroom.

Run on RunPod GPU:  python3 -m tests.test_selective_load
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, BitsAndBytesConfig  # noqa: E402

from engine.model import prune_to_layers  # noqa: E402
from engine.stage import stage_for_range  # noqa: E402
from engine.kv import StageKV  # noqa: E402

MODEL = os.environ.get("CIRCUIT_QUANT_MODEL", "Qwen/Qwen2.5-3B-Instruct")


def main():
    print(f"selective-load test — {MODEL} (4-bit, stage = second half of layers)")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb,
                                                 device_map="cuda").eval()
    nL = model.config.num_hidden_layers
    H = model.config.hidden_size
    vram_full = torch.cuda.memory_allocated() / 1e9
    print(f"  full model: {nL} layers, {vram_full:.2f} GB")

    start, end = nL // 2, nL                       # this stage owns the second half
    prune_to_layers(model, start, end, keep_head=False)
    vram_pruned = torch.cuda.memory_allocated() / 1e9
    print(f"  pruned to layers [{start},{end}): {vram_pruned:.2f} GB "
          f"({100*(vram_full-vram_pruned)/vram_full:.0f}% freed)")

    # the pruned stage must still run a forward
    stage = stage_for_range(model, start, end)
    cache = StageKV(model.config)
    h = torch.randn(1, 5, H, dtype=torch.float16, device="cuda")
    pos = torch.arange(5, device="cuda").unsqueeze(0)
    with torch.no_grad():
        out = stage.forward(h, pos, past_key_values=cache, use_cache=True)
    ok_shape = tuple(out.shape) == (1, 5, H)
    print(f"  pruned stage forward: out shape {tuple(out.shape)} ok={ok_shape} "
          f"kv_len={cache.get_seq_length()}")
    assert ok_shape and cache.get_seq_length() == 5, "pruned stage forward broken"
    assert vram_pruned < vram_full * 0.75, "pruning did not free meaningful VRAM"
    print("SELECTIVE LOAD OK — stage holds only its layers, VRAM freed, forward works")


if __name__ == "__main__":
    main()
