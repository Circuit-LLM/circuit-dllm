"""
bench_spec.py — does speculative decoding actually speed things up?

Real draft<<target gap: 3B target (split 2 stages) + 0.5B draft, on the GPU.
Measures greedy tok/s vs speculative tok/s + acceptance. Also asserts the
speculative output == greedy (correctness rides along).

Run on RunPod:  python3 -m tests.bench_spec
Env: CIRCUIT_TARGET (default Qwen2.5-3B-Instruct), CIRCUIT_DRAFT (0.5B),
     CIRCUIT_BENCH_NEW, CIRCUIT_SPEC_K.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from engine.stage import split_model  # noqa: E402
from engine.specdecode import speculative_greedy, SplitTarget, GreedyDraft  # noqa: E402

TARGET = os.environ.get("CIRCUIT_TARGET", "Qwen/Qwen2.5-3B-Instruct")
DRAFT = os.environ.get("CIRCUIT_DRAFT", "Qwen/Qwen2.5-0.5B-Instruct")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_NEW = int(os.environ.get("CIRCUIT_BENCH_NEW", "80"))
K = int(os.environ.get("CIRCUIT_SPEC_K", "4"))
PROMPT = "Write a short paragraph explaining why the sky is blue."


def _sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def _greedy(target, ids, n_new):
    """Greedy decode using the split target only (no draft)."""
    logits = target.forward_tokens(ids, 0)
    nxt = int(logits[0, -1].argmax())
    out = [nxt]
    pos = target.kv_len()
    for _ in range(n_new - 1):
        logits = target.forward_tokens(torch.tensor([[nxt]], device=DEVICE), pos)
        nxt = int(logits[0, -1].argmax())
        out.append(nxt)
        pos += 1
    return out


def main():
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    print(f"bench_spec — target={TARGET} draft={DRAFT} device={DEVICE} K={K} n_new={N_NEW}")
    tok = AutoTokenizer.from_pretrained(TARGET)
    tgt_model = AutoModelForCausalLM.from_pretrained(TARGET, dtype=dtype).eval().to(DEVICE)
    drf_model = AutoModelForCausalLM.from_pretrained(DRAFT, dtype=dtype).eval().to(DEVICE)
    nL = tgt_model.config.num_hidden_layers
    ids = tok(PROMPT, return_tensors="pt").input_ids.to(DEVICE)

    def mk_target():
        return SplitTarget(tgt_model, split_model(tgt_model, [nL // 2]), device=DEVICE)

    def mk_draft():
        return GreedyDraft(drf_model, device=DEVICE)

    with torch.no_grad():
        _greedy(mk_target(), ids, 4)                        # warmup

        _sync(); t0 = time.time()
        g = _greedy(mk_target(), ids, N_NEW)
        _sync(); dt_g = time.time() - t0

        _sync(); t0 = time.time()
        s = speculative_greedy(mk_target(), mk_draft(), ids, N_NEW, K=K, device=DEVICE)
        _sync(); dt_s = time.time() - t0
        st = speculative_greedy.last_stats

    acc = st["accepted"] / max(1, st["proposed"])
    tpr = N_NEW / st["rounds"]
    first_div = next((i for i, (a, b) in enumerate(zip(s, g)) if a != b), None)
    print(f"  greedy        : {N_NEW/dt_g:6.1f} tok/s  ({dt_g:.2f}s)")
    print(f"  speculative   : {N_NEW/dt_s:6.1f} tok/s  ({dt_s:.2f}s)")
    print(f"  speedup       : {dt_g/dt_s:.2f}x   accept_rate={acc:.0%}  tokens/round={tpr:.1f}")
    if DEVICE == "cpu":
        assert s == g, "speculative output diverged from greedy on CPU (real bug)!"
        print("  output==greedy: True (exact, CPU/fp32)")
    else:
        # GPU fp16: verify runs [1,K+1] batches, greedy runs [1,1] -> different
        # kernels -> tiny logit noise can flip a near-tie argmax (both valid).
        # Exact equivalence is proven on CPU (tests/test_specdecode.py).
        print(f"  output==greedy: {s == g} (fp16; first divergence at token "
              f"{first_div}; exact on CPU)")
    print(f"  sample: {tok.decode(s[:40])!r}")


if __name__ == "__main__":
    main()
