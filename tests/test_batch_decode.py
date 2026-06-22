"""
test_batch_decode.py — Win B / B1: a batched forward is token-identical to N
sequential forwards.

The correctness foundation of intra-step batching. B sequences of DIFFERENT prompt
lengths are decoded together as one left-padded batch through the SAME stage path
(embed -> stages with per-stage KV -> norm -> head), and every sequence's tokens must
match running it ALONE. This is what proves the ragged-KV machinery — the 2D padding
mask folded into the causal mask + per-row position_ids driving RoPE — is exact. A
mask/index bug here is a SILENT wrong-output, so this test is the gate for everything
after it. In-process, no sockets, CPU/fp32 for determinism.

Run:  python3 -m tests.test_batch_decode
"""

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from engine.stage import split_model  # noqa: E402
from engine.kv import StageKV  # noqa: E402

MODEL = os.environ.get("CIRCUIT_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
N_NEW = 24
# deliberately different lengths -> the batch is ragged, exercising padding/masking
PROMPTS = [
    "The capital of France is",
    "Two plus two equals",
    "The largest planet in the solar system is the gas giant",
    "Water",
]


def seq_decode(model, stages, ids, n_new):
    """Reference: decode ONE sequence alone through the stage path (no padding)."""
    caches = [StageKV(model.config) for _ in stages]
    pos = torch.arange(ids.shape[1]).unsqueeze(0)
    h = model.model.embed_tokens(ids)
    for st, c in zip(stages, caches):
        h = st.forward(h, pos, past_key_values=c, use_cache=True)
    nxt = model.lm_head(model.model.norm(h))[:, -1].argmax(-1, keepdim=True)
    out = [int(nxt)]
    cur = ids.shape[1]
    for _ in range(n_new - 1):
        h = model.model.embed_tokens(nxt)
        p = torch.tensor([[cur]])
        for st, c in zip(stages, caches):
            h = st.forward(h, p, past_key_values=c, use_cache=True)
        nxt = model.lm_head(model.model.norm(h))[:, -1].argmax(-1, keepdim=True)
        out.append(int(nxt))
        cur += 1
    return out


def batch_decode(model, stages, ids, attn, n_new):
    """Decode B sequences together as one left-padded batch. ids/attn: [B, max_P]
    (left-padded; attn 1=real, 0=pad). Returns a list of token lists, one per row."""
    B = ids.shape[0]
    caches = [StageKV(model.config) for _ in stages]
    # left-pad position_ids: real tokens carry their true positions, pads -> 0 (masked)
    pos = (attn.long().cumsum(-1) - 1).clamp(min=0)
    h = model.model.embed_tokens(ids)
    for st, c in zip(stages, caches):
        h = st.forward(h, pos, past_key_values=c, use_cache=True, attention_mask=attn)
    # last physical slot is the last REAL token for every row (right-aligned prefill)
    nxt = model.lm_head(model.model.norm(h))[:, -1].argmax(-1, keepdim=True)   # [B,1]
    lengths = attn.long().sum(-1)                                              # [B] real lengths
    out = [[int(nxt[b])] for b in range(B)]
    cur_attn = attn
    for _ in range(n_new - 1):
        cur_attn = torch.cat([cur_attn, torch.ones(B, 1, dtype=attn.dtype)], dim=1)
        step_pos = lengths.unsqueeze(1)                                        # [B,1] per-row pos
        h = model.model.embed_tokens(nxt)                                     # [B,1,D]
        for st, c in zip(stages, caches):
            h = st.forward(h, step_pos, past_key_values=c, use_cache=True, attention_mask=cur_attn)
        nxt = model.lm_head(model.model.norm(h))[:, -1].argmax(-1, keepdim=True)
        lengths = lengths + 1
        for b in range(B):
            out[b].append(int(nxt[b]))
    return out


def main():
    print(f"B1 batch-decode test — model={MODEL}, B={len(PROMPTS)}, n_new={N_NEW}")
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    stages = split_model(model, [model.config.num_hidden_layers // 2])

    # per-sequence references (each decoded alone)
    refs = [seq_decode(model, stages, tok(p, return_tensors="pt").input_ids, N_NEW)
            for p in PROMPTS]

    # one ragged left-padded batch
    enc = tok(PROMPTS, return_tensors="pt", padding=True)
    lens = enc.attention_mask.sum(-1).tolist()
    print(f"  prompt token lengths (ragged): {lens}  -> padded to {enc.input_ids.shape[1]}")
    batched = batch_decode(model, stages, enc.input_ids, enc.attention_mask, N_NEW)

    allok = True
    for b, p in enumerate(PROMPTS):
        ok = batched[b] == refs[b]
        allok = allok and ok
        print(f"  [{'OK ' if ok else 'BAD'}] seq {b} (len {lens[b]}): {tok.decode(batched[b])!r}")
        if not ok:
            print(f"        ref: {tok.decode(refs[b])!r}")
            print(f"        got tokens: {batched[b]}")
            print(f"        ref tokens: {refs[b]}")
    assert allok, "batched decode diverged from sequential — ragged mask/positions are WRONG"
    print("B1 PASSED — ragged batched decode is token-identical to sequential, for every row")


if __name__ == "__main__":
    main()
