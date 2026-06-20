"""
probe_cudagraph.py — does CUDA-graphing the draft decode step actually help?

Compares single-token decode tok/s for the 0.5B draft: eager vs
torch.compile(mode="reduce-overhead") (CUDA graphs) + StaticCache (fixed
addresses, required for graph capture). This is the draft accelerator that makes
speculative decode win once per-call overhead is gone.

Run on RunPod GPU:  python3 -m tests.probe_cudagraph
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache  # noqa: E402

MODEL = os.environ.get("CIRCUIT_DRAFT", "Qwen/Qwen2.5-0.5B-Instruct")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N = int(os.environ.get("CIRCUIT_BENCH_NEW", "128"))
MAXLEN = 1024


def decode_loop(model, ids, n, dtype, use_static):
    """Greedy decode n tokens. Returns (tokens, seconds)."""
    if use_static:
        cache = StaticCache(config=model.config, max_batch_size=1,
                            max_cache_len=MAXLEN, device=DEVICE, dtype=dtype)
    else:
        from transformers.cache_utils import DynamicCache
        cache = DynamicCache(config=model.config)

    L = ids.shape[1]
    cp = torch.arange(L, device=DEVICE)
    logits = model(input_ids=ids, past_key_values=cache, cache_position=cp).logits
    nxt = logits[:, -1].argmax(-1, keepdim=True)
    out = [int(nxt)]
    torch.cuda.synchronize() if DEVICE == "cuda" else None
    t0 = time.time()
    for i in range(n - 1):
        cp = torch.tensor([L + i], device=DEVICE)
        logits = model(input_ids=nxt, past_key_values=cache, cache_position=cp).logits
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        out.append(int(nxt))
    torch.cuda.synchronize() if DEVICE == "cuda" else None
    return out, time.time() - t0


def main():
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    print(f"probe_cudagraph — {MODEL} device={DEVICE} n={N}")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype).eval().to(DEVICE)
    ids = tok("Once upon a time", return_tensors="pt").input_ids.to(DEVICE)

    with torch.no_grad():
        _, dt_eager = decode_loop(model, ids, N, dtype, use_static=False)
        print(f"  eager (dynamic cache) : {N/dt_eager:6.1f} tok/s  ({dt_eager:.2f}s)")

        cmodel = torch.compile(model, mode="reduce-overhead", fullgraph=True)
        decode_loop(cmodel, ids, 8, dtype, use_static=True)        # warmup / capture
        _, dt_cg = decode_loop(cmodel, ids, N, dtype, use_static=True)
        print(f"  compiled (CUDA graph) : {N/dt_cg:6.1f} tok/s  ({dt_cg:.2f}s)")
        print(f"  speedup               : {dt_eager/dt_cg:.2f}x")


if __name__ == "__main__":
    main()
