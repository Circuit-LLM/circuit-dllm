"""
test_awq_subslice_gpu.py — validate AWQ-per-node pre-sliced sub-checkpoints (docs/AWQ_PER_NODE.md).

THE make-or-break question for mesh lever #2: does gptqmodel LOAD a pre-sliced n-layer AWQ
sub-checkpoint, and do its layers produce the SAME output as the full model's corresponding
layers? If yes, each mesh node can run AWQ/Marlin on its slice (2.1× bnb) by loading a complete
small AWQ model — sidestepping the AWQ-can't-shard blocker.

Runs on a GPU with a small AWQ model (default Qwen2.5-7B-Instruct-AWQ) so it's cheap/fast; the
result generalizes to the 72B (same gptqmodel load path, same Stage forward).

What it checks:
  1. LOAD: slice layers [start,end) → a sub-model dir; AutoModelForCausalLM loads it on cuda
     (Marlin post_init sees only cuda layers → should pass). A hard failure here IS the answer.
  2. CORRECTNESS: the sub-model's local layers 0..n-1, run via Stage with GLOBAL layer-indices
     (RoPE depends on position not layer index), match the full model's stage_for_range(start,end)
     on the same hidden state — within fp16 tolerance (ideally bitwise, same weights+kernel).

    python3 -m tests.test_awq_subslice_gpu [model_id] [start] [end]
"""
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch  # noqa: E402
from transformers import DynamicCache  # noqa: E402

from engine.model import load_model  # noqa: E402
from engine.stage import Stage, stage_for_range  # noqa: E402


def _slice(full_dir, out_dir, start, end, keep_norm=False):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "slice_awq", os.path.join(REPO, "scripts", "slice-awq.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    argv = [full_dir, out_dir, str(start), str(end)] + (["--keep-norm"] if keep_norm else [])
    rc = m.main(argv)
    if rc != 0:
        raise RuntimeError(f"slice-awq failed rc={rc}")


def main():
    model_id = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-7B-Instruct-AWQ"
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    end = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    dev = "cuda"
    assert torch.cuda.is_available(), "needs a GPU"

    from huggingface_hub import snapshot_download
    print(f"resolving {model_id} …", flush=True)
    full_dir = snapshot_download(model_id)
    print(f"full checkpoint: {full_dir}", flush=True)

    out_dir = os.path.join(tempfile.gettempdir(), f"awq-sub-{start}-{end}")
    _slice(full_dir, out_dir, start, end)

    # 1. LOAD the full model + the sub-model (the make-or-break gptqmodel question)
    print("loading FULL model on cuda …", flush=True)
    full = load_model(model_id, device=dev)
    n_full = full.config.num_hidden_layers
    assert 0 <= start < end <= n_full, f"range [{start},{end}) outside [0,{n_full})"

    print(f"loading SUB model ({end-start} layers) on cuda …", flush=True)
    try:
        sub = load_model(out_dir, device=dev)
    except Exception as e:
        print(f"\n*** SUB-MODEL LOAD FAILED — AWQ-per-node needs the fallback ***\n{type(e).__name__}: {e}")
        print("Fallbacks (docs/AWQ_PER_NODE.md): load as AutoModel (no CausalLM head), or keep "
              "lm_head/embed in the slice. Re-run after adjusting the slicer.")
        raise
    assert sub.config.num_hidden_layers == end - start, "sub config layer count"

    # 2a. DEFINITIVE: the sliced layers' weight tensors are BYTEWISE identical to the full
    # model's layers [start,end). This is deterministic (no fp16 noise) — if every quantized
    # buffer (qweight/scales/qzeros, repacked by Marlin) matches, the slice changed nothing.
    def _tensors(mod):
        d = {n: t for n, t in mod.named_buffers()}
        d.update({n: t for n, t in mod.named_parameters()})
        return d
    mism = []
    for i in range(end - start):
        L, S = _tensors(full.model.layers[start + i]), _tensors(sub.model.layers[i])
        keys = set(L) | set(S)
        for k in sorted(keys):
            if k not in L or k not in S:
                mism.append(f"layer{i}.{k}: present in only one"); continue
            a, b = L[k], S[k]
            if a.shape != b.shape:
                mism.append(f"layer{i}.{k}: shape {tuple(a.shape)} vs {tuple(b.shape)}")
            elif not torch.equal(a, b):
                mism.append(f"layer{i}.{k}: max|Δ|={ (a.float()-b.float()).abs().max():.3e}")
    n_tensors = sum(len(_tensors(sub.model.layers[i])) for i in range(end - start))
    if mism:
        print(f"\n*** WEIGHT MISMATCH ({len(mism)}/{n_tensors}) — slicing corrupted weights ***")
        for m in mism[:12]:
            print("   ", m)
        raise AssertionError("sliced sub-model weights differ from the full model's layers")
    print(f"\nweights: ALL {n_tensors} tensors across {end-start} layers bytewise-identical "
          f"to full layers [{start},{end})", flush=True)

    # 2b. forward equivalence with a NONDETERMINISM CONTROL: run the full-model stage twice to
    # measure the fp16/Marlin run-to-run floor, then compare sub-vs-full against that floor.
    hidden = full.config.hidden_size
    T = 12
    torch.manual_seed(0)
    h = torch.randn(1, T, hidden, dtype=torch.float16, device=dev)
    pos = torch.arange(T, device=dev)[None]
    ref_stage = stage_for_range(full, start, end)
    sub_stage = Stage(list(sub.model.layers), list(range(start, end)),
                      sub.model.rotary_emb, full.config)

    def _fwd(stage):
        return stage.forward(h.clone(), pos, past_key_values=DynamicCache(), use_cache=True)
    out_ref1, out_ref2, out_sub = _fwd(ref_stage), _fwd(ref_stage), _fwd(sub_stage)
    scale = out_ref1.float().abs().max().item() + 1e-9
    floor = (out_ref1.float() - out_ref2.float()).abs().max().item()   # full-vs-full noise floor
    diff = (out_ref1.float() - out_sub.float()).abs().max().item()     # sub-vs-full
    print(f"forward: sub-vs-full max|Δ|={diff:.3e} (rel {diff/scale:.2e}); "
          f"full-vs-full floor={floor:.3e}; hidden-scale={scale:.1f}", flush=True)

    # weights identical ⇒ any forward diff is pure kernel nondeterminism; require it within the
    # measured floor (+ a small epsilon) AND relatively tiny.
    assert diff <= max(floor * 1.5, 1e-3) and diff / scale < 5e-3, (
        f"sub-vs-full {diff:.3e} exceeds the nondeterminism floor {floor:.3e} — unexpected")
    print(f"\nAWQ-PER-NODE VALIDATED — gptqmodel loads the pre-sliced n-layer sub-checkpoint on "
          f"cuda (Marlin OK); its layers are bytewise-identical to the full model and produce "
          f"the same output within the fp16 noise floor. Pre-slicing works → each node can run "
          f"AWQ/Marlin on its slice.")


if __name__ == "__main__":
    main()
