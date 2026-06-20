"""
test_specdecode_stream.py — STREAMING speculative decode == plain greedy.

Same invariant as test_specdecode.py, but exercises the generator used by the
live API (speculative_greedy_stream): the yielded token sequence must equal
plain greedy for ANY draft, and EOS must stop generation cleanly.

Run on RunPod (CPU/fp32 for determinism):  python3 -m tests.test_specdecode_stream
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from transformers.cache_utils import DynamicCache  # noqa: E402

from engine.stage import split_model  # noqa: E402
from engine.specdecode import speculative_greedy_stream, SplitTarget, GreedyDraft  # noqa: E402

MODEL = os.environ.get("CIRCUIT_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
PROMPT = "The capital of France is"
N_NEW = 24
K = 4


def _plain_greedy(model, ids, n_new):
    cache = DynamicCache(config=model.config)
    nxt = model(ids, past_key_values=cache, use_cache=True).logits[:, -1].argmax(-1, keepdim=True)
    out = [int(nxt)]
    for _ in range(n_new - 1):
        nxt = model(nxt, past_key_values=cache, use_cache=True).logits[:, -1].argmax(-1, keepdim=True)
        out.append(int(nxt))
    return out


def _stream(model, ids, perturb, eos_ids=()):
    stages = split_model(model, [model.config.num_hidden_layers // 2])
    target = SplitTarget(model, stages, device="cpu")
    draft = GreedyDraft(model, device="cpu")
    return list(speculative_greedy_stream(target, draft, ids, N_NEW, eos_ids=eos_ids,
                                          K=K, device="cpu", draft_perturb=perturb))


def main():
    print(f"specdecode-stream test — model={MODEL} K={K} n_new={N_NEW}")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    ids = tok(PROMPT, return_tensors="pt").input_ids
    ref = _plain_greedy(model, ids, N_NEW)

    cases = [
        ("accept-all", None),
        ("all-reject", lambda i, t: (t + 1) % 1000),
        ("mixed",      lambda i, t: (t + 1) % 1000 if i % 2 else t),
    ]
    for name, perturb in cases:
        got = _stream(model, ids, perturb)
        ok = got == ref
        print(f"  [{name:10s}] match={ok}  tokens={len(got)}")
        if not ok:
            print("     ref:", tok.decode(ref))
            print("     got:", tok.decode(got))
        assert ok, f"streaming speculative diverged from greedy in case '{name}'"

    # EOS stop: injecting ref[6] as an EOS id must stop right before it.
    eos_got = _stream(model, ids, None, eos_ids=(ref[6],))
    assert eos_got == ref[:6], f"EOS stop wrong: {eos_got} vs {ref[:6]}"
    print(f"  [eos-stop  ] match=True  stopped at {len(eos_got)} tokens (EOS not emitted)")

    print("ALL STREAMING SPECULATIVE CHECKS PASSED  (stream == greedy; EOS stops cleanly)")


if __name__ == "__main__":
    main()
