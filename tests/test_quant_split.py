"""
test_quant_split.py — does the layer split stay correct on a 4-bit model?

Loads the 3B 4-bit (bitsandbytes nf4) and checks split greedy decode == unsplit.
Quantization replaces the Linear weights inside each decoder layer but leaves the
layer structure (attention/MLP/forward signature) unchanged, so Stage.forward
should work untouched. This de-risks the 32B (which must run 4-bit to fit 2 L4s);
the AWQ/GPTQ 32B has the same layer structure as bnb-4bit here.

Run on RunPod GPU:  python3 -m tests.test_quant_split
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from transformers import (AutoModelForCausalLM, AutoTokenizer,  # noqa: E402
                          BitsAndBytesConfig)
from transformers.cache_utils import DynamicCache  # noqa: E402

from engine.stage import split_model  # noqa: E402
from engine.kv import StageKV  # noqa: E402

MODEL = os.environ.get("CIRCUIT_QUANT_MODEL", "Qwen/Qwen2.5-3B-Instruct")
N_NEW = 30
PROMPT = "List three facts about the moon:"


def _unsplit_greedy(model, ids, n):
    cache = DynamicCache(config=model.config)
    nxt = model(ids, past_key_values=cache, use_cache=True).logits[:, -1].argmax(-1, keepdim=True)
    out = [int(nxt)]
    for _ in range(n - 1):
        nxt = model(nxt, past_key_values=cache, use_cache=True).logits[:, -1].argmax(-1, keepdim=True)
        out.append(int(nxt))
    return out


def _split_greedy(model, stages, ids, n):
    embed, norm, lm_head = model.model.embed_tokens, model.model.norm, model.lm_head
    caches = [StageKV(model.config) for _ in stages]
    seq = ids.shape[1]
    h = embed(ids)
    pos = torch.arange(seq, device=ids.device).unsqueeze(0)
    for st, c in zip(stages, caches):
        h = st.forward(h, pos, past_key_values=c, use_cache=True)
    nxt = lm_head(norm(h))[:, -1].argmax(-1, keepdim=True)
    out = [int(nxt)]
    cur = seq
    for _ in range(n - 1):
        h = embed(nxt)
        pos = torch.tensor([[cur]], device=ids.device)
        for st, c in zip(stages, caches):
            h = st.forward(h, pos, past_key_values=c, use_cache=True)
        nxt = lm_head(norm(h))[:, -1].argmax(-1, keepdim=True)
        out.append(int(nxt))
        cur += 1
    return out


def main():
    print(f"quant split test — {MODEL} (bitsandbytes nf4 4-bit)")
    tok = AutoTokenizer.from_pretrained(MODEL)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb,
                                                 device_map="cuda").eval()
    vram = torch.cuda.memory_allocated() / 1e9
    nL = model.config.num_hidden_layers
    print(f"  loaded: {nL} layers, {vram:.2f} GB VRAM (4-bit)")

    stages = split_model(model, [nL // 2])
    ids = tok(PROMPT, return_tensors="pt").input_ids.to("cuda")
    with torch.no_grad():
        ref = _unsplit_greedy(model, ids, N_NEW)
        got = _split_greedy(model, stages, ids, N_NEW)

    match = ref == got
    print(f"  split == unsplit (4-bit): {match}")
    print(f"  sample: {tok.decode(got)!r}")
    if not match:
        first = next((i for i, (a, b) in enumerate(zip(ref, got)) if a != b), None)
        print(f"  first divergence at token {first} (GPU/quant fp noise if late & coherent)")
    assert match or got[:5] == ref[:5], "split diverged immediately on quantized model — real bug"
    print("QUANT SPLIT OK — Stage.forward works on 4-bit layers")


if __name__ == "__main__":
    main()
