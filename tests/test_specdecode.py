"""
test_specdecode.py — greedy speculative decode == plain greedy, for ANY draft.

  [1] accept-all : draft == target  -> ~100% acceptance, output == greedy
  [2] all-reject : draft always wrong -> every token corrected, output == greedy
  [3] mixed      : draft wrong every 2nd token -> partial accept, output == greedy

The invariant under test: the draft changes only speed, never the result.
Run on RunPod (CPU/fp32 for determinism):  python3 -m tests.test_specdecode
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from transformers.cache_utils import DynamicCache  # noqa: E402

from engine.stage import split_model  # noqa: E402
from engine.specdecode import speculative_greedy, SplitTarget, GreedyDraft  # noqa: E402

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


def _run(model, ids, perturb):
    stages = split_model(model, [model.config.num_hidden_layers // 2])
    target = SplitTarget(model, stages, device="cpu")
    draft = GreedyDraft(model, device="cpu")
    got = speculative_greedy(target, draft, ids, N_NEW, K=K, device="cpu",
                             draft_perturb=perturb)
    return got, speculative_greedy.last_stats


def main():
    print(f"specdecode test — model={MODEL} K={K} n_new={N_NEW}")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    ids = tok(PROMPT, return_tensors="pt").input_ids
    ref = _plain_greedy(model, ids, N_NEW)

    cases = [
        ("accept-all", None),
        ("all-reject", lambda i, t: (t + 1) % 1000),          # never the target's pick
        ("mixed",      lambda i, t: (t + 1) % 1000 if i % 2 else t),
    ]
    for name, perturb in cases:
        got, stats = _run(model, ids, perturb)
        ok = got == ref
        acc = stats["accepted"] / max(1, stats["proposed"])
        print(f"  [{name:10s}] match={ok}  accept_rate={acc:4.0%}  "
              f"rounds={stats['rounds']} (tokens/round={N_NEW/stats['rounds']:.1f})")
        if not ok:
            print("     ref:", tok.decode(ref))
            print("     got:", tok.decode(got))
        assert ok, f"speculative output diverged from greedy in case '{name}'"

    print("ALL SPECULATIVE-DECODE CHECKS PASSED  (output == greedy for every draft)")


if __name__ == "__main__":
    main()
