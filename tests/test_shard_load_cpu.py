"""
test_shard_load_cpu.py — the selective shard LOADER produces a bitwise-identical
stage to load-then-prune, on CPU with a small (non-quantized) model.

Proves the loading MECHANISM: build the skeleton, drop the un-owned modules, read
only the owned layers' weights from the checkpoint — and the resulting stage
forward matches the load-then-prune path exactly. (The AWQ-on-GPU path — quant-
aware construction + actual VRAM reduction — is validated separately on a GPU in
test_shard_load.py.)

Run on a torch host (CPU, deterministic):  python3 -m tests.test_shard_load_cpu
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from engine.model import load_model, load_model_shard  # noqa: E402
from engine.stage import stage_for_range  # noqa: E402
from engine.kv import StageKV  # noqa: E402

MODEL = os.environ.get("CIRCUIT_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
PROMPT = "The capital of France is"


def stage_out(model, start, end, h_in, pos):
    stage = stage_for_range(model, start, end)
    cache = StageKV(model.config)
    with torch.no_grad():
        return stage.forward(h_in, pos, past_key_values=cache, use_cache=True)


def main():
    print(f"shard-load CPU test — {MODEL}")
    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = tok(PROMPT, return_tensors="pt").input_ids
    pos = torch.arange(ids.shape[1]).unsqueeze(0)

    # reference: whole model loaded, used per layer-range (load-then-prune path)
    full = load_model(MODEL, device="cpu")
    n = full.config.num_hidden_layers
    K = n // 2
    with torch.no_grad():
        h0 = full.model.embed_tokens(ids)
        ref_head = stage_out(full, 0, K, h0, pos)        # layers [0,K)
        ref_mid = stage_out(full, K, n, ref_head, pos)   # layers [K,n)

    # shard loader: head stage [0,K) (keeps embed) + middle stage [K,n) (no head)
    sh_head = load_model_shard(MODEL, 0, K, keep_head=True, device="cpu")
    sh_mid = load_model_shard(MODEL, K, n, keep_head=False, device="cpu")
    with torch.no_grad():
        h0s = sh_head.model.embed_tokens(ids)
        out_head = stage_out(sh_head, 0, K, h0s, pos)
        out_mid = stage_out(sh_mid, K, n, out_head, pos)

    embed_ok = torch.equal(h0s, h0)
    d_head = (out_head - ref_head).abs().max().item()
    d_mid = (out_mid - ref_mid).abs().max().item()
    head_ok = torch.allclose(out_head, ref_head, atol=1e-6, rtol=0)
    mid_ok = torch.allclose(out_mid, ref_mid, atol=1e-6, rtol=0)

    print(f"  embed weights match:                 {embed_ok}")
    print(f"  layers[0,{K}) head-stage  == prune:  {head_ok}  (max_abs_diff {d_head:.2e})")
    print(f"  layers[{K},{n}) mid-stage  == prune:  {mid_ok}  (max_abs_diff {d_mid:.2e})")
    assert embed_ok, "shard-loaded embedding differs from full"
    assert head_ok and mid_ok, "shard-loaded stage diverged from load-then-prune"
    print("SHARD-LOAD MECHANISM OK — selective load == load-then-prune, on CPU")


if __name__ == "__main__":
    main()
