"""
test_correctness.py — Phase 0 DoD: a model split across stages is bitwise
identical to the single-process reference.

Checks:
  1. PREFILL    : split logits == model(input_ids).logits   (bitwise)
  2. WIRE       : same, but the hidden state crosses pack/unpack_activation
  3. DECODE+KV  : greedy decode with per-stage KV == model.generate greedy

Run on RunPod (needs transformers + a model download). CPU + fp32 for
determinism:  python3 -m tests.test_correctness
Env: CIRCUIT_TEST_MODEL (default Qwen/Qwen2.5-0.5B-Instruct), CIRCUIT_TEST_DEVICE.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from transformers.cache_utils import DynamicCache  # noqa: E402

from engine.stage import split_model  # noqa: E402
from engine.kv import StageKV  # noqa: E402
from engine import tensors as T  # noqa: E402

MODEL_ID = os.environ.get("CIRCUIT_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
DEVICE = os.environ.get("CIRCUIT_TEST_DEVICE", "cpu")
PROMPT = "The capital of France is"


def _load():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    model.eval().to(DEVICE)
    return tok, model


def _coord_parts(model):
    inner = model.model
    return inner.embed_tokens, inner.norm, model.lm_head


def _prefill(model, stages, input_ids, via_wire=False):
    embed, norm, lm_head = _coord_parts(model)
    hidden = embed(input_ids)
    seq = input_ids.shape[1]
    position_ids = torch.arange(seq, device=input_ids.device).unsqueeze(0)
    for i, st in enumerate(stages):
        hidden = st.forward(hidden, position_ids, past_key_values=None, use_cache=False)
        if via_wire and i < len(stages) - 1:
            # serialize the inter-stage hidden state exactly as the wire would
            _, _, _, hidden = T.unpack_activation(T.pack_activation(1, seq, hidden))
            hidden = hidden.to(DEVICE)
    return lm_head(norm(hidden))


def _decode_greedy(model, stages, input_ids, n_new):
    # Each stage keeps its OWN per-stage KV (StageKV) — the real distributed
    # design, where stages live on different machines and cannot share a cache.
    embed, norm, lm_head = _coord_parts(model)
    caches = [StageKV(model.config) for _ in stages]
    hidden = embed(input_ids)
    seq = input_ids.shape[1]
    pos = torch.arange(seq, device=input_ids.device).unsqueeze(0)
    for st, c in zip(stages, caches):
        hidden = st.forward(hidden, pos, past_key_values=c, use_cache=True)
    nxt = lm_head(norm(hidden))[:, -1].argmax(-1, keepdim=True)
    out = [nxt]
    cur = seq
    for _ in range(n_new - 1):
        h = embed(nxt)
        pos = torch.tensor([[cur]], device=input_ids.device)
        for st, c in zip(stages, caches):
            h = st.forward(h, pos, past_key_values=c, use_cache=True)
        nxt = lm_head(norm(h))[:, -1].argmax(-1, keepdim=True)
        out.append(nxt)
        cur += 1
    return torch.cat(out, dim=1)


def _decode_greedy_ref(model, input_ids, n_new):
    """Unsplit pure-greedy argmax decode with one cache — the apples-to-apples
    reference. (model.generate applies the model's generation_config, incl.
    repetition_penalty=1.1, so it is NOT a pure-greedy reference.)"""
    cache = DynamicCache(config=model.config)
    logits = model(input_ids, past_key_values=cache, use_cache=True).logits
    nxt = logits[:, -1].argmax(-1, keepdim=True)
    out = [nxt]
    for _ in range(n_new - 1):
        logits = model(nxt, past_key_values=cache, use_cache=True).logits
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        out.append(nxt)
    return torch.cat(out, dim=1)


def main():
    print(f"correctness harness — model={MODEL_ID} device={DEVICE} dtype=fp32")
    tok, model = _load()
    n = model.config.num_hidden_layers
    boundary = n // 2
    stages = split_model(model, [boundary])
    print(f"  {n} layers split {boundary}/{n - boundary} across 2 stages")
    ids = tok(PROMPT, return_tensors="pt").input_ids.to(DEVICE)

    # 1. PREFILL bitwise
    with torch.no_grad():
        ref = model(ids, use_cache=False).logits
        got = _prefill(model, stages, ids)
    bitwise = torch.equal(ref, got)
    maxdiff = (ref - got).abs().max().item()
    print(f"  [1] prefill: bitwise={bitwise} max_abs_diff={maxdiff:.3e}")
    assert bitwise, f"prefill not bitwise-identical (max diff {maxdiff})"

    # 2. WIRE round-trip between stages
    with torch.no_grad():
        got_w = _prefill(model, stages, ids, via_wire=True)
    assert torch.equal(ref, got_w), "wire path altered the result"
    print("  [2] wire: hidden state crosses pack/unpack -> still bitwise-identical")

    # 3. DECODE + KV: split (per-stage StageKV) vs unsplit pure-greedy reference
    n_new = 20
    with torch.no_grad():
        ref_new = _decode_greedy_ref(model, ids, n_new)
        got_new = _decode_greedy(model, stages, ids, n_new)
    match = torch.equal(ref_new, got_new)
    print(f"  [3] decode+KV (per-stage): {match} ({n_new} greedy tokens)")
    if not match:
        print("      ref:", tok.decode(ref_new[0]))
        print("      got:", tok.decode(got_new[0]))
    assert match, "split per-stage KV decode diverged from unsplit greedy"

    print("ALL CORRECTNESS CHECKS PASSED")
    print("  sample:", repr(tok.decode(got_new[0])))


if __name__ == "__main__":
    main()
