"""
test_slice_awq.py — the pure index/config remap core of scripts/slice-awq.py
(docs/AWQ_PER_NODE.md). No torch / no safetensors: deterministic dict/string surgery.

Covers: layer renumbering to 0-based, dropping un-owned layers + embed/lm_head, keep_norm,
config rewrite (num_hidden_layers + tie_word_embeddings), the _kept_tensor_names ↔
remap_weight_map agreement that the I/O path asserts, and range validation.

    python3 -m tests.test_slice_awq
"""
import importlib.util
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# load scripts/slice-awq.py (hyphenated → not importable as a normal module)
_spec = importlib.util.spec_from_file_location(
    "slice_awq", os.path.join(REPO, "scripts", "slice-awq.py"))
slice_awq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(slice_awq)

remap_weight_map = slice_awq.remap_weight_map
sub_config = slice_awq.sub_config
_kept_tensor_names = slice_awq._kept_tensor_names


def _full_map(n_layers, awq=True):
    """A synthetic full-model weight_map like an AWQ safetensors index: embed, lm_head, norm,
    and per-layer AWQ tensors (qweight/qzeros/scales for the attn/mlp projections)."""
    wm = {
        "model.embed_tokens.weight": "s0",
        "model.norm.weight": "s9",
        "lm_head.weight": "s9",
    }
    projs = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
             "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
    tens = (["qweight", "qzeros", "scales"] if awq else ["weight"])
    for i in range(n_layers):
        shard = f"s{i // 10}"
        for p in projs:
            for t in tens:
                wm[f"model.layers.{i}.{p}.{t}"] = shard
        wm[f"model.layers.{i}.input_layernorm.weight"] = shard
        wm[f"model.layers.{i}.post_attention_layernorm.weight"] = shard
    return wm


def main():
    wm = _full_map(80)

    # ── slice [16,48): 32 local layers, renumbered 0..31 ──────────────────────
    out = remap_weight_map(wm, 16, 48)
    # global layer 16 → local 0; layer 47 → local 31; 48 excluded
    assert "model.layers.0.self_attn.q_proj.qweight" in out, "global 16 → local 0"
    assert "model.layers.31.mlp.down_proj.scales" in out, "global 47 → local 31"
    assert not any(k.startswith("model.layers.32.") for k in out), "local stops at 31 (no 32)"
    # only decoder-layer tensors survive by default — embed/lm_head/norm dropped
    assert "model.embed_tokens.weight" not in out, "embed dropped (stage runs on hidden states)"
    assert "lm_head.weight" not in out, "lm_head dropped (no logit projection on a stage)"
    assert "model.norm.weight" not in out, "norm dropped by default"
    # local layer count is exactly end-start
    local_layers = {int(k.split(".")[2]) for k in out if k.startswith("model.layers.")}
    assert local_layers == set(range(32)), f"expected local 0..31, got {sorted(local_layers)[:3]}…"
    # AWQ tensors preserved (the whole point — Marlin loads these)
    assert sum(k.endswith(".qweight") for k in out) == 32 * 7, "7 AWQ projections × 32 layers"

    # ── keep_norm re-adds model.norm (e.g. a last-stage that norms locally) ────
    out_n = remap_weight_map(wm, 48, 80, keep_norm=True)
    assert "model.norm.weight" in out_n, "keep_norm preserves the final norm"
    assert "lm_head.weight" not in out_n, "keep_norm still drops lm_head"

    # ── keep_head: coordinator slice keeps embed + norm + lm_head ─────────────
    out_h = remap_weight_map(wm, 0, 16, keep_head=True)
    assert "model.embed_tokens.weight" in out_h, "keep_head preserves embed"
    assert "lm_head.weight" in out_h, "keep_head preserves lm_head"
    assert "model.norm.weight" in out_h, "keep_head preserves norm"
    assert {int(k.split(".")[2]) for k in out_h if k.startswith("model.layers.")} == set(range(16))
    sc_h = sub_config({"num_hidden_layers": 80, "tie_word_embeddings": True}, 0, 16, keep_head=True)
    assert sc_h["tie_word_embeddings"] is True, "keep_head keeps original tie (has real head)"
    assert _kept_tensor_names(wm.keys(), 0, 16, keep_norm=False, keep_head=True).get(
        "lm_head.weight") == "lm_head.weight", "keep_head kept-names includes lm_head"

    # ── _kept_tensor_names mirrors remap (the I/O copy ↔ index agreement) ─────
    # this is the invariant main() asserts: the originals it copies, renamed, == the new index.
    kept = _kept_tensor_names(wm.keys(), 16, 48, keep_norm=False)
    assert set(kept.values()) == set(remap_weight_map(wm, 16, 48)), \
        "_kept_tensor_names renamed set must equal remap_weight_map keys"

    # ── sub_config: num_hidden_layers shrinks, tie forced off, quant preserved ─
    cfg = {"num_hidden_layers": 80, "hidden_size": 8192, "tie_word_embeddings": True,
           "quantization_config": {"quant_method": "awq", "bits": 4}}
    sc = sub_config(cfg, 16, 48)
    assert sc["num_hidden_layers"] == 32, "sub-model has 32 layers"
    assert sc["tie_word_embeddings"] is False, "no lm_head to tie → forced false"
    assert sc["quantization_config"]["quant_method"] == "awq", "AWQ path preserved"
    assert sc["hidden_size"] == 8192 and cfg["num_hidden_layers"] == 80, \
        "other fields copied; original config not mutated"

    # ── range validation ──────────────────────────────────────────────────────
    for bad in ((48, 16), (-1, 4), (5, 5)):
        try:
            remap_weight_map(wm, *bad); raise AssertionError(f"should reject range {bad}")
        except ValueError:
            pass
    # a valid-shaped range that selects nothing (beyond the model) also raises
    try:
        remap_weight_map(_full_map(8), 16, 24); raise AssertionError("should reject empty selection")
    except ValueError:
        pass

    # ── works for non-AWQ (plain .weight) maps too — pure structural surgery ───
    out_fp = remap_weight_map(_full_map(80, awq=False), 0, 4)
    assert {int(k.split(".")[2]) for k in out_fp if k.startswith("model.layers.")} == set(range(4))

    print("SLICE-AWQ CORE TESTS PASSED — layer renumber, embed/lm_head/norm drop, keep_norm, "
          "kept↔remap agreement, sub_config rewrite, range validation")


if __name__ == "__main__":
    main()
