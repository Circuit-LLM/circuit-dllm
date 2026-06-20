"""
test_kv.py — StageKV speculative-rollback correctness.

Proves truncate_to(L) restores a cache to its exact state at length L: a decode
step run after rollback produces a bitwise-identical hidden state to the same
step run before rollback. This is the operation speculative decode relies on
when it rejects guessed tokens.

Run on RunPod:  python3 -m tests.test_kv
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from engine.stage import stage_for_range  # noqa: E402
from engine.kv import StageKV  # noqa: E402

MODEL = os.environ.get("CIRCUIT_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
PROMPT = "Once upon a time"


def main():
    print(f"kv rollback test — model={MODEL}")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    n = model.config.num_hidden_layers
    stage = stage_for_range(model, 0, n)  # whole model as one stage
    embed = model.model.embed_tokens

    ids = tok(PROMPT, return_tensors="pt").input_ids
    L = ids.shape[1]
    cache = StageKV(model.config)

    with torch.no_grad():
        # prefill -> cache length L
        stage.forward(embed(ids), torch.arange(L).unsqueeze(0),
                      past_key_values=cache, use_cache=True)
        assert cache.get_seq_length() == L, cache.get_seq_length()

        # a fixed "next token" hidden to decode at position L
        nxt = torch.tensor([[123]])
        h_in = embed(nxt)
        pos = torch.tensor([[L]])

        # decode once (before rollback)
        out_before = stage.forward(h_in, pos, past_key_values=cache, use_cache=True)
        assert cache.get_seq_length() == L + 1

        # roll the speculative token back
        cache.truncate_to(L)
        assert cache.get_seq_length() == L, f"truncate left len {cache.get_seq_length()}"

        # decode the same token again (after rollback)
        out_after = stage.forward(h_in, pos, past_key_values=cache, use_cache=True)

    same = torch.equal(out_before, out_after)
    print(f"  rollback restores exact state: {same}")
    assert same, "decode after truncate_to differs -> rollback did not restore state"
    print("KV ROLLBACK TEST PASSED")


if __name__ == "__main__":
    main()
