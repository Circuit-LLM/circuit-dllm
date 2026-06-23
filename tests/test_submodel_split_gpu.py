"""
test_submodel_split_gpu.py — end-to-end AWQ-per-node pipeline correctness (docs/AWQ_PER_NODE.md).

Proves a full split greedy decode built from PRE-SLICED AWQ sub-models is token-identical to the
single full model:
  • coordinator slice [0,k) WITH --keep-head (embed + norm + lm_head + layers 0..k-1)
  • stage slice [k,N) (pure: layers only, renumbered 0..)
  pipeline:  embed(tok) → coord layers[0,k) → stage layers[k,N) → norm → lm_head → argmax
This is the in-process numerical proof for the 72B mesh (the socket transport is already proven
for the bnb path); it also validates the coordinator's keep-head slice loads + serves embed/lm_head.

    python3 -m tests.test_submodel_split_gpu [model_id] [k]
"""
import importlib.util
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch  # noqa: E402
from transformers import AutoTokenizer, DynamicCache  # noqa: E402

from engine.model import load_model  # noqa: E402
from engine.stage import Stage, stage_for_range  # noqa: E402

_spec = importlib.util.spec_from_file_location("slice_awq", os.path.join(REPO, "scripts", "slice-awq.py"))
slice_awq = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(slice_awq)


def _slice(full_dir, out_dir, s, e, keep_head=False):
    argv = [full_dir, out_dir, str(s), str(e)] + (["--keep-head"] if keep_head else [])
    if slice_awq.main(argv) != 0:
        raise RuntimeError("slice failed")


def _decode(embed, stages, norm, lm_head, prompt_ids, n_new, dev):
    """Greedy decode n_new tokens through embed → stages (each a Stage) → norm → lm_head, with a
    per-stage DynamicCache (mirrors the coordinator's per-stage KV)."""
    caches = [DynamicCache() for _ in stages]
    cur, pos0, out = prompt_ids, 0, []
    for _ in range(n_new):
        T = cur.shape[1]
        pos = torch.arange(pos0, pos0 + T, device=dev)[None]
        h = embed(cur)
        for st, c in zip(stages, caches):
            h = st.forward(h, pos, past_key_values=c, use_cache=True)
        nxt = lm_head(norm(h)[:, -1:]).argmax(-1)   # [1,1]
        out.append(int(nxt)); cur = nxt; pos0 += T
    return out


def main():
    model_id = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-7B-Instruct-AWQ"
    dev = "cuda"
    assert torch.cuda.is_available(), "needs a GPU"
    from huggingface_hub import snapshot_download
    full_dir = snapshot_download(model_id)

    full = load_model(model_id, device=dev)
    N = full.config.num_hidden_layers
    k = int(sys.argv[2]) if len(sys.argv) > 2 else N // 2
    print(f"{model_id}: {N} layers, split at {k} → coord[0,{k}) + stage[{k},{N})", flush=True)

    coord_dir = os.path.join(tempfile.gettempdir(), f"sub-coord-0-{k}")
    stage_dir = os.path.join(tempfile.gettempdir(), f"sub-stage-{k}-{N}")
    _slice(full_dir, coord_dir, 0, k, keep_head=True)
    _slice(full_dir, stage_dir, k, N, keep_head=False)

    coord = load_model(coord_dir, device=dev)     # has embed/norm/lm_head + layers 0..k-1
    stg = load_model(stage_dir, device=dev)        # pure: layers (global k..N-1) renumbered 0..
    assert coord.config.num_hidden_layers == k and stg.config.num_hidden_layers == N - k

    tok = AutoTokenizer.from_pretrained(full_dir)
    prompt = "Quantum computing is"
    ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
    n_new = 24

    # reference: the full model as one stage [0,N)
    ref = _decode(full.model.embed_tokens, [stage_for_range(full, 0, N)],
                  full.model.norm, full.lm_head, ids, n_new, dev)
    # split: coord head + coord layers[0,k) then stage layers[k,N); each Stage uses LOCAL indices
    coord_stage = Stage(list(coord.model.layers), list(range(k)), coord.model.rotary_emb, coord.config)
    stage_stage = Stage(list(stg.model.layers), list(range(N - k)), stg.model.rotary_emb, stg.config)
    sub = _decode(coord.model.embed_tokens, [coord_stage, stage_stage],
                  coord.model.norm, coord.lm_head, ids, n_new, dev)

    print("ref:", tok.decode(ref), flush=True)
    print("sub:", tok.decode(sub), flush=True)
    match = sum(a == b for a, b in zip(ref, sub))
    print(f"\ntoken match: {match}/{n_new}", flush=True)
    assert ref == sub, (f"split pipeline diverged from full model at token "
                        f"{next(i for i,(a,b) in enumerate(zip(ref,sub)) if a!=b)} — NOT correct")
    print(f"\nAWQ-PER-NODE PIPELINE VALIDATED — a {N}-layer greedy decode split as a keep-head "
          f"coordinator slice [0,{k}) + a pure stage slice [{k},{N}) is TOKEN-IDENTICAL to the "
          f"full model over {n_new} tokens. The pre-sliced AWQ mesh is numerically correct.")


if __name__ == "__main__":
    main()
