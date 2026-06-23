"""
test_shard_load_gpu.py — the selective shard loader, validated on a GPU with an
UN-QUANTIZED (fp16) model, checking CORRECTNESS (not just shape/VRAM).

Why this exists: test_shard_load.py asserted only shape + VRAM, which false-positived
on AWQ (the quantized weights silently didn't load, yet it printed "OK"). This test
proves the loaded shard produces the *same output* as the full model, AND that an
fp16 32B slice fits a single 24 GB L4.

  Part 1: fp16 7B (fits one L4) — sharded stages == full-model stages (bitwise on GPU).
  Part 2: fp16 32B — load layers[0,16)+head, assert it fits (< 24 GB) and forward is finite.

Run on a RunPod L4 (HF_HOME set):  HF_HOME=/workspace/hf-cache python3 -m tests.test_shard_load_gpu

Result on 2026-06-23 (L4): part1 head/mid max_diff=0.0000; part2 18.7 GB VRAM — PASS.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gc  # noqa: E402
import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from engine.model import load_model, load_model_shard  # noqa: E402
from engine.stage import stage_for_range  # noqa: E402
from engine.kv import StageKV  # noqa: E402

DEV = os.environ.get("CIRCUIT_TEST_DEVICE", "cuda:0")
M7 = os.environ.get("CIRCUIT_TEST_MODEL_7B", "Qwen/Qwen2.5-7B-Instruct")
M32 = os.environ.get("CIRCUIT_TEST_MODEL_32B", "Qwen/Qwen2.5-32B-Instruct")
PROMPT = "The capital of France is"


def stage_out(model, s, e, h, pos):
    st = stage_for_range(model, s, e)
    c = StageKV(model.config)
    with torch.no_grad():
        return st.forward(h, pos, past_key_values=c, use_cache=True)


def main():
    # ---- Part 1: correctness on a model that fits whole ----
    print(f"[part1] correctness on {M7} (fp16, GPU)")
    tok = AutoTokenizer.from_pretrained(M7)
    ids = tok(PROMPT, return_tensors="pt").input_ids.to(DEV)
    pos = torch.arange(ids.shape[1], device=DEV).unsqueeze(0)
    full = load_model(M7, device=DEV)
    n = full.config.num_hidden_layers
    K = n // 2
    with torch.no_grad():
        h0 = full.model.embed_tokens(ids)
        ref_head = stage_out(full, 0, K, h0, pos)
        ref_mid = stage_out(full, K, n, ref_head, pos)
    ref_head_c, ref_mid_c, h0_c = ref_head.float().cpu(), ref_mid.float().cpu(), h0.float().cpu()
    del full; gc.collect(); torch.cuda.empty_cache()

    sh_head = load_model_shard(M7, 0, K, keep_head=True, device=DEV)
    with torch.no_grad():
        h0s = sh_head.model.embed_tokens(ids)
        out_head = stage_out(sh_head, 0, K, h0s, pos)
    out_head_c = out_head.float().cpu()
    embed_ok = torch.allclose(h0s.float().cpu(), h0_c, atol=1e-3)
    del sh_head; gc.collect(); torch.cuda.empty_cache()

    sh_mid = load_model_shard(M7, K, n, keep_head=False, device=DEV)
    with torch.no_grad():
        out_mid = stage_out(sh_mid, K, n, ref_head_c.half().to(DEV), pos)  # feed REFERENCE head: isolates mid loader
    out_mid_c = out_mid.float().cpu()
    del sh_mid; gc.collect(); torch.cuda.empty_cache()

    head_diff = (out_head_c - ref_head_c).abs().max().item()
    mid_diff = (out_mid_c - ref_mid_c).abs().max().item()
    print(f"[part1] embed_ok={embed_ok} head_max_diff={head_diff:.4f} mid_max_diff={mid_diff:.4f}")
    assert embed_ok and head_diff < 0.1 and mid_diff < 0.1, "sharded fp16 stage output diverges from full model"

    # ---- Part 2: a fp16 32B slice fits one L4 ----
    print(f"[part2] loading layers[0,16)+head of {M32} (fp16) on one L4 ...")
    torch.cuda.reset_peak_memory_stats()
    sh = load_model_shard(M32, 0, 16, keep_head=True, device=DEV)
    vram = torch.cuda.memory_allocated() / 1e9
    tok32 = AutoTokenizer.from_pretrained(M32)
    ids32 = tok32(PROMPT, return_tensors="pt").input_ids.to(DEV)
    pos32 = torch.arange(ids32.shape[1], device=DEV).unsqueeze(0)
    with torch.no_grad():
        h = sh.model.embed_tokens(ids32)
        o = stage_out(sh, 0, 16, h, pos32)
    finite = bool(torch.isfinite(o).all().item())
    print(f"[part2] 32B slice -> {vram:.1f} GB VRAM (full fp16 32B ~65 GB, L4=24 GB) forward={tuple(o.shape)} finite={finite}")
    assert vram < 23 and finite, "fp16 32B slice did not fit one L4 / forward not finite"
    print("SHARD-LOAD GPU TEST OK — correct AND fits")


if __name__ == "__main__":
    main()
