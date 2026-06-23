#!/usr/bin/env python3
"""
slice-awq.py — slice an AWQ checkpoint into a per-stage sub-model (docs/AWQ_PER_NODE.md).

Each stage of the mesh loads a SELF-CONSISTENT n-layer AWQ model (every layer on cuda →
gptqmodel's Marlin post_init is happy), instead of one 80-layer checkpoint with most layers
on meta (which Marlin rejects). This dodges the AWQ-can't-shard blocker so each node runs
AWQ/Marlin (2.1× bnb) on its slice.

Two parts:
  • the PURE remap core (remap_weight_map / sub_config) — stdlib only, unit-tested offline in
    tests/test_slice_awq.py. Deterministic index/config surgery, no torch / no safetensors.
  • main() — pod-side I/O that reads the full checkpoint and writes the sub-model. Lazy-imports
    safetensors so importing this module for the pure core needs nothing.

Usage (on a pod with the full AWQ checkpoint staged):
    python3 scripts/slice-awq.py <full_ckpt_dir> <out_dir> <start> <end> [--keep-norm]
e.g. python3 scripts/slice-awq.py /root/hf-cache/...Qwen2.5-72B-Instruct-AWQ \
        /root/stage-A 16 48
"""
from __future__ import annotations

import json
import os
import re
import sys
from typing import Dict, Tuple

# Tensor-name prefixes for the transformer decoder layers across HF Qwen2/Llama-style models.
_LAYER_RE = re.compile(r"^(model\.layers\.)(\d+)(\..*)$")


def _relayer_name(name: str, start: int, end: int) -> str | None:
    """Map a full-model tensor name to its sub-model name, or None to DROP it.
    Decoder layers in [start,end) are renumbered to 0-based; layers outside are dropped.
    Non-layer tensors (embed_tokens, norm, lm_head, rotary) are dropped here — keep_* in
    remap_weight_map decides which non-layer tensors to re-add."""
    m = _LAYER_RE.match(name)
    if not m:
        return None                      # non-layer tensor → handled by remap_weight_map
    idx = int(m.group(2))
    if not (start <= idx < end):
        return None                      # un-owned layer → drop
    return f"{m.group(1)}{idx - start}{m.group(3)}"


def _keep_nonlayer(name: str, keep_norm: bool, keep_head: bool) -> bool:
    """Which non-decoder-layer tensors a slice keeps. A pure STAGE keeps none (runs on hidden
    states). The COORDINATOR's slice keeps the head parts (keep_head): embed_tokens (embeds the
    prompt), model.norm + lm_head (final logits). keep_norm alone keeps just the final norm."""
    if keep_head and (name.startswith("model.embed_tokens.") or name.startswith("lm_head.")
                      or name.startswith("model.norm.")):
        return True
    if keep_norm and name.startswith("model.norm."):
        return True
    return False


def remap_weight_map(weight_map: Dict[str, str], start: int, end: int,
                     keep_norm: bool = False, keep_head: bool = False) -> Dict[str, str]:
    """Filter a safetensors index `weight_map` (tensor-name → shard-file) down to a sub-model:
    keep decoder layers [start,end) renumbered to 0..n-1, plus the non-layer tensors selected by
    keep_norm / keep_head (see _keep_nonlayer). A pure STAGE keeps layers only (runs on hidden
    states, never embeds or projects to logits); the COORDINATOR uses keep_head to also hold
    embed/norm/lm_head. Pure: stdlib only. Returns the new {sub_name: shard_file}.

    Raises ValueError if the requested range has no layers in the map (caught a typo'd range)."""
    if not (0 <= start < end):
        raise ValueError(f"bad range [{start},{end})")
    out: Dict[str, str] = {}
    seen_layer = False
    for name, shard in weight_map.items():
        sub = _relayer_name(name, start, end)
        if sub is not None:
            out[sub] = shard
            seen_layer = True
        elif _keep_nonlayer(name, keep_norm, keep_head):
            out[name] = shard
    if not seen_layer:
        raise ValueError(f"range [{start},{end}) selected no decoder layers from the index "
                         f"(check the layer count / prefix)")
    return out


def sub_config(config: dict, start: int, end: int, keep_head: bool = False) -> dict:
    """The sub-model's config.json: the full config verbatim except num_hidden_layers = end-start.
    A pure stage has no lm_head to tie → tie_word_embeddings forced false; a keep_head slice keeps
    the original tie setting (it has the real embed/lm_head). quantization_config is preserved so
    the loader takes the AWQ/Marlin path. Pure."""
    c = dict(config)
    c["num_hidden_layers"] = end - start
    if not keep_head:
        c["tie_word_embeddings"] = False
    return c


def _kept_tensor_names(all_names, start, end, keep_norm, keep_head=False):
    """The set of FULL-model tensor names a slice keeps (for the I/O copy). Mirrors
    remap_weight_map's keep rules but returns originals (so the reader knows what to pull)."""
    kept = {}
    for name in all_names:
        sub = _relayer_name(name, start, end)
        if sub is not None:
            kept[name] = sub
        elif _keep_nonlayer(name, keep_norm, keep_head):
            kept[name] = name
    return kept


def main(argv):
    if len(argv) < 4:
        print(__doc__); return 2
    full_dir, out_dir, start, end = argv[0], argv[1], int(argv[2]), int(argv[3])
    flags = argv[4:]
    keep_norm = "--keep-norm" in flags
    keep_head = "--keep-head" in flags   # coordinator slice: also keep embed + norm + lm_head
    os.makedirs(out_dir, exist_ok=True)

    # lazy imports — only needed for the actual copy, not for the pure core / unit tests
    from safetensors import safe_open
    from safetensors.torch import save_file

    idx_path = os.path.join(full_dir, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            index = json.load(f)
        weight_map = index["weight_map"]
    else:
        # single-shard checkpoint: synthesize a weight_map over the one file
        only = "model.safetensors"
        with safe_open(os.path.join(full_dir, only), framework="pt") as h:
            weight_map = {k: only for k in h.keys()}

    kept = _kept_tensor_names(weight_map.keys(), start, end, keep_norm, keep_head)
    print(f"slicing layers [{start},{end}) → {end-start} local layers; "
          f"{len(kept)} tensors kept (keep_norm={keep_norm}, keep_head={keep_head})", flush=True)

    # group originals by their shard file so each shard is opened once
    by_shard: Dict[str, list] = {}
    for orig in kept:
        by_shard.setdefault(weight_map[orig], []).append(orig)

    new_tensors = {}
    for shard, names in by_shard.items():
        with safe_open(os.path.join(full_dir, shard), framework="pt", device="cpu") as h:
            for orig in names:
                new_tensors[kept[orig]] = h.get_tensor(orig)

    out_weights = os.path.join(out_dir, "model.safetensors")
    save_file(new_tensors, out_weights, metadata={"format": "pt"})
    # write a single-file index so the loader finds every tensor in model.safetensors
    new_map = remap_weight_map(weight_map, start, end, keep_norm=keep_norm, keep_head=keep_head)
    assert set(new_map) == set(new_tensors), "index/tensor mismatch — remap core disagrees with I/O"
    with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": {k: "model.safetensors" for k in new_map}}, f)

    with open(os.path.join(full_dir, "config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(sub_config(cfg, start, end, keep_head=keep_head), f, indent=2)

    # copy tokenizer + generation config if present (harmless; some loaders look for them)
    for aux in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                "generation_config.json", "special_tokens_map.json"):
        src = os.path.join(full_dir, aux)
        if os.path.exists(src):
            with open(src, "rb") as a, open(os.path.join(out_dir, aux), "wb") as b:
                b.write(a.read())
    print(f"wrote sub-model → {out_dir} ({end-start} layers, {len(new_tensors)} tensors)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
