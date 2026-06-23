"""
shard_fetch.py — a node's "get my AWQ slice" resolver (docs/AWQ_PER_NODE.md, productization #1/#2).

In the dynamic mesh a node registers, the coordinator assigns it a layer range [start,end), and
the node must obtain a COMPLETE n-layer AWQ sub-model for exactly that range to serve it (Marlin
on cuda — the AWQ-per-node win). resolve_submodel() returns a local dir holding that sub-model,
trying three sources in order — cheapest first:

  1. LOCAL CACHE   — a sub-model dir we already produced this run (re-register → same range → reuse)
  2. PUBLISHED ARTIFACT — download just this range (~16GB) from a Circuit-LLM HF repo (the
     contributor on-ramp: pull your slice, not the full 40GB). Needs the artifact published
     (scripts/publish-awq-shards.py) for this exact range.
  3. LOCAL SLICE   — slice [start,end) out of a staged full AWQ checkpoint (scripts/slice-awq.py).
     Always works if the node has the full checkpoint; the fallback when no artifact is published.

Pure helpers (slot_dirname / artifact_subdir) are unit-tested; the I/O paths shell out to the
proven slicer CLI and huggingface_hub.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys


# ── pipeline layout / catalog (shared by the publisher AND the coordinator) ───
# A "layout" is the canonical contiguous slot partition of a model: the FIRST slice is the
# coordinator's keep-head slice (embed+norm+lm_head + layers [0,c)), the rest are pure stage
# slices. The publisher (publish-awq-shards.py) slices artifacts for these exact ranges, and the
# coordinator builds its Topology from the SAME layout (topology_from_catalog) — so every slot the
# coordinator assigns matches a published artifact and a joining node DOWNLOADS its slice instead
# of slicing the full 40GB locally. One definition, used both places → ranges can't drift.

def parse_layout(spec: str):
    """'0:59,59:80' -> [(0,59,True),(59,80,False)]: contiguous ranges, the FIRST flagged keep_head
    (the coordinator slice). Validates contiguity + non-empty/ascending ranges. Pure."""
    parts = [p for p in spec.split(",") if p.strip()]
    if not parts:
        raise ValueError("empty layout")
    ranges, prev_end = [], None
    for i, p in enumerate(parts):
        s_str, e_str = p.split(":")
        s, e = int(s_str), int(e_str)
        if e <= s:
            raise ValueError(f"range {p}: end must be > start")
        if prev_end is not None and s != prev_end:
            raise ValueError(f"non-contiguous layout at {p}: expected start {prev_end}")
        ranges.append((s, e, i == 0))
        prev_end = e
    return ranges


def build_manifest(model_id: str, ranges) -> dict:
    """Manifest a node reads to find its slot's artifact. Pure."""
    return {
        "model": model_id,
        "format": "awq-per-node-v1",
        "num_layers": ranges[-1][1],
        "slots": [
            {"start": s, "end": e, "keep_head": kh, "dir": slot_dirname(s, e, kh)}
            for (s, e, kh) in ranges
        ],
    }


def catalog_layout(spec) -> list:
    """Normalize a catalog into [(start,end,keep_head)]: accepts a layout STRING ('0:59,59:80'),
    a manifest DICT, or a path to a manifest.json. The coordinator and the node both resolve their
    layout through this so they agree on slot boundaries."""
    if isinstance(spec, dict):
        return [(s["start"], s["end"], s["keep_head"]) for s in spec["slots"]]
    if isinstance(spec, str) and (spec.endswith(".json") or os.path.isfile(spec)):
        with open(spec) as f:
            return [(s["start"], s["end"], s["keep_head"]) for s in json.load(f)["slots"]]
    return parse_layout(spec)


def topology_from_catalog(num_layers: int, ranges):
    """Map a layout to the coordinator's Topology args: the first slice is the coordinator's
    co-located keep-head range [0,coordinator_end); the rest are the mesh stage slots. Returns
    (coordinator_end, num_stages, slot_sizes). Validates the stages tile [coordinator_end,num_layers)."""
    if not ranges or not ranges[0][2]:
        raise ValueError("catalog's first slot must be the coordinator keep-head slice")
    coordinator_end = ranges[0][1]
    stages = ranges[1:]
    if not stages:
        raise ValueError("catalog has no stage slots (coordinator-only not a mesh)")
    if stages[0][0] != coordinator_end or stages[-1][1] != num_layers:
        raise ValueError(f"stage slots must tile [{coordinator_end},{num_layers}), got {stages}")
    return coordinator_end, len(stages), [e - s for (s, e, _kh) in stages]


def slot_dirname(start: int, end: int, keep_head: bool = False) -> str:
    """Canonical local dir / artifact name for a slot. Coordinator (keep_head) slices start at 0
    and carry embed+norm+lm_head, so they get a distinct name from a pure stage of the same range."""
    return f"{'coord' if keep_head else 'sub'}-{start}-{end}"


def artifact_subdir(start: int, end: int, keep_head: bool = False) -> str:
    """Path of a slot's artifact within a published repo (one subfolder per slot)."""
    return slot_dirname(start, end, keep_head)


def _has_weights(d: str) -> bool:
    return os.path.isfile(os.path.join(d, "model.safetensors"))


def _engine_dir(engine_dir):
    return engine_dir or os.environ.get("CIRCUIT_ENGINE_DIR") or os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))


def _try_download(repo: str, start: int, end: int, keep_head: bool, dest: str, log) -> bool:
    """Download just this slot's artifact subfolder from `repo` and point `dest` at it. Returns
    True only if `dest` ends up with weights; False to fall through to local slicing."""
    sub = artifact_subdir(start, end, keep_head)
    try:
        from huggingface_hub import snapshot_download
        path = snapshot_download(repo, allow_patterns=[f"{sub}/*"])
        src = os.path.join(path, sub)
        if not _has_weights(src):
            log("INFO", "artifact repo has no such slot — will slice locally", repo=repo, slot=sub)
            return False
        if os.path.abspath(src) != os.path.abspath(dest):
            if os.path.islink(dest) or os.path.exists(dest):
                try: os.remove(dest)
                except OSError: pass
            try:
                os.symlink(src, dest)              # cheap: no 16GB copy, point at the HF cache
            except OSError:
                return False                        # can't link (cross-fs) → slice locally instead
        log("INFO", "downloaded published slice", repo=repo, slot=sub)
        return _has_weights(dest)
    except Exception as e:
        log("INFO", "artifact download failed — will slice locally", repo=repo, slot=sub, err=str(e)[:120])
        return False


def _local_slice(model_id: str, start: int, end: int, keep_head: bool, dest: str, engine_dir: str, log):
    """Slice [start,end) out of the staged full AWQ checkpoint via scripts/slice-awq.py."""
    from huggingface_hub import snapshot_download
    full = snapshot_download(model_id)
    cmd = [sys.executable, os.path.join(engine_dir, "scripts", "slice-awq.py"), full, dest,
           str(start), str(end)] + (["--keep-head"] if keep_head else [])
    log("INFO", "slicing locally from full checkpoint", model=model_id, slot=f"{start}:{end}", keep_head=keep_head)
    subprocess.run(cmd, check=True)


def resolve_submodel(model_id: str, start: int, end: int, work_dir: str = "/root", *,
                     keep_head: bool = False, repo: str | None = None, engine_dir: str | None = None,
                     log=None) -> str:
    """Return a local dir holding the AWQ sub-model for layers [start,end) of model_id, sourcing it
    (in order) from: local cache → published artifact `repo` → local slice of the full checkpoint.
    `repo` None/'' → skip the download path (slice locally). `keep_head` for the coordinator slice."""
    if log is None:
        def log(*a, **k): pass
    engine_dir = _engine_dir(engine_dir)
    dest = os.path.join(work_dir, slot_dirname(start, end, keep_head))
    if _has_weights(dest):
        log("INFO", "using cached slice", dir=dest)
        return dest
    os.makedirs(work_dir, exist_ok=True)
    if repo and _try_download(repo, start, end, keep_head, dest, log) and _has_weights(dest):
        return dest
    _local_slice(model_id, start, end, keep_head, dest, engine_dir, log)
    if not _has_weights(dest):
        raise RuntimeError(f"failed to resolve sub-model for [{start},{end}) -> {dest}")
    return dest
