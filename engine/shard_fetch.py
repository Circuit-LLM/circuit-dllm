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

import os
import subprocess
import sys


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
