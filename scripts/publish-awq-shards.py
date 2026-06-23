#!/usr/bin/env python3
"""
publish-awq-shards.py — slice a full AWQ checkpoint into per-stage artifacts + a manifest and
publish them to a Circuit-LLM HF repo (docs/AWQ_PER_NODE.md, productization #1).

The contributor on-ramp: instead of every node downloading the full 40GB AWQ to slice locally,
a one-time job slices the canonical pipeline layout into per-slot sub-models (each ~16GB), uploads
them to one repo (one subfolder per slot), and writes a manifest. A joining node then pulls ONLY
its assigned slot via engine.shard_fetch.resolve_submodel(repo=...).

Layout: a comma list of start:end ranges; the FIRST is the coordinator's keep-head slice (embed +
norm + lm_head + layers [0,k)), the rest are pure stage slices. Get the bandwidth-proportional
layout for a target fleet from `python3 -m engine.topology layout <N> <gpu0> <gpu1> ...`.

Usage:
  python3 scripts/publish-awq-shards.py <full_awq_repo_or_dir> <out_dir> --layout 0:59,59:80 \
      [--repo Circuit-LLM/qwen2.5-72b-awq-shards] [--upload]
  # --upload requires HF_TOKEN with write access to the repo. Without it, just stages locally.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
# layout/manifest live in engine.shard_fetch so the publisher and the coordinator share ONE
# definition of the slot boundaries (catalog alignment — see topology_from_catalog).
from engine.shard_fetch import slot_dirname, parse_layout, build_manifest  # noqa: E402


def main(argv):
    if len(argv) < 2 or "--layout" not in argv:
        print(__doc__); return 2
    full, out_dir = argv[0], argv[1]
    rest = argv[2:]
    layout = rest[rest.index("--layout") + 1]
    repo = rest[rest.index("--repo") + 1] if "--repo" in rest else None
    do_upload = "--upload" in rest
    ranges = parse_layout(layout)
    os.makedirs(out_dir, exist_ok=True)

    from huggingface_hub import snapshot_download
    full_dir = full if os.path.isdir(full) else snapshot_download(full)

    for (s, e, kh) in ranges:
        dest = os.path.join(out_dir, slot_dirname(s, e, kh))
        if os.path.isfile(os.path.join(dest, "model.safetensors")):
            print(f"[publish] {dest} exists, skip"); continue
        cmd = [sys.executable, os.path.join(REPO_ROOT, "scripts", "slice-awq.py"),
               full_dir, dest, str(s), str(e)] + (["--keep-head"] if kh else [])
        print(f"[publish] slicing {s}:{e} keep_head={kh} -> {dest}")
        subprocess.run(cmd, check=True)

    manifest = build_manifest(full if not os.path.isdir(full) else "(local)", ranges)
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[publish] wrote manifest: {json.dumps(manifest)}")

    if do_upload:
        if not repo:
            print("[publish] --upload needs --repo"); return 2
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(repo, private=True, exist_ok=True)
        api.upload_folder(folder_path=out_dir, repo_id=repo)
        print(f"[publish] uploaded {out_dir} -> {repo} (private)")
    else:
        print(f"[publish] staged locally at {out_dir} (pass --upload --repo <id> to publish)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
