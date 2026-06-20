"""
bench_split.py — GPU baseline: greedy decode tok/s for the split vs single-process.

Confirms the split runs on CUDA and gives the Phase 1 baseline (synchronous, no
speculation yet). Output is just sanity-checked for coherence — bitwise
correctness is proven separately on CPU/fp32 (test_correctness).

Run on RunPod:  python3 -m tests.bench_split
Env: CIRCUIT_TEST_MODEL, CIRCUIT_BENCH_NEW (tokens), CIRCUIT_BENCH_DEVICE.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from engine.stage import split_model  # noqa: E402
from engine.kv import StageKV  # noqa: E402

MODEL = os.environ.get("CIRCUIT_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
DEVICE = os.environ.get("CIRCUIT_BENCH_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
N_NEW = int(os.environ.get("CIRCUIT_BENCH_NEW", "100"))
PROMPT = "Explain how a neural network learns, step by step:"


def _sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def _greedy_single(model, tok, n_new):
    from transformers.cache_utils import DynamicCache
    ids = tok(PROMPT, return_tensors="pt").input_ids.to(DEVICE)
    cache = DynamicCache(config=model.config)
    logits = model(ids, past_key_values=cache, use_cache=True).logits
    nxt = logits[:, -1].argmax(-1, keepdim=True)
    out = [int(nxt)]
    for _ in range(n_new - 1):
        nxt = model(nxt, past_key_values=cache, use_cache=True).logits[:, -1].argmax(-1, keepdim=True)
        out.append(int(nxt))
    return out


def _greedy_split(model, tok, stages, n_new):
    embed, norm, lm_head = model.model.embed_tokens, model.model.norm, model.lm_head
    ids = tok(PROMPT, return_tensors="pt").input_ids.to(DEVICE)
    seq = ids.shape[1]
    caches = [StageKV(model.config) for _ in stages]
    h = embed(ids)
    pos = torch.arange(seq, device=DEVICE).unsqueeze(0)
    for st, c in zip(stages, caches):
        h = st.forward(h, pos, past_key_values=c, use_cache=True)
    nxt = lm_head(norm(h))[:, -1].argmax(-1, keepdim=True)
    out = [int(nxt)]
    cur = seq
    for _ in range(n_new - 1):
        h = embed(nxt)
        pos = torch.tensor([[cur]], device=DEVICE)
        for st, c in zip(stages, caches):
            h = st.forward(h, pos, past_key_values=c, use_cache=True)
        nxt = lm_head(norm(h))[:, -1].argmax(-1, keepdim=True)
        out.append(int(nxt))
        cur += 1
    return out


def _timed(fn, *a):
    _sync(); t0 = time.time()
    out = fn(*a)
    _sync(); dt = time.time() - t0
    return out, dt


def main():
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    print(f"bench — model={MODEL} device={DEVICE} dtype={dtype} n_new={N_NEW}")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype).eval().to(DEVICE)
    n = model.config.num_hidden_layers
    stages = split_model(model, [n // 2])

    with torch.no_grad():
        _greedy_split(model, tok, stages, 4)          # warmup
        out_s, dt_s = _timed(_greedy_single, model, tok, N_NEW)
        out_p, dt_p = _timed(_greedy_split, model, tok, stages, N_NEW)

    print(f"  single-process : {N_NEW/dt_s:6.1f} tok/s  ({dt_s:.2f}s)")
    print(f"  2-stage split  : {N_NEW/dt_p:6.1f} tok/s  ({dt_p:.2f}s)")
    print(f"  split overhead : {100*(dt_p-dt_s)/dt_s:+.1f}%  (in-process; WAN/async not in play yet)")
    print(f"  sample: {tok.decode(out_p[:24])!r}")


if __name__ == "__main__":
    main()
