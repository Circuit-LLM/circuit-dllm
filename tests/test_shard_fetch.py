"""
test_shard_fetch.py — AWQ-per-node distribution logic (docs/AWQ_PER_NODE.md, productization #1/#2).
Pure/offline parts: slot naming, the node's resolve_submodel CACHE path, layout parsing, manifest.
(The download + local-slice paths need HF/torch → validated on a pod.)

    python3 -m tests.test_shard_fetch
"""
import importlib.util
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from engine.shard_fetch import slot_dirname, artifact_subdir, resolve_submodel  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "publish_awq", os.path.join(REPO, "scripts", "publish-awq-shards.py"))
pub = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(pub)


def main():
    # ── slot naming: coordinator (keep-head) vs pure stage are distinct ───────
    assert slot_dirname(16, 48) == "sub-16-48"
    assert slot_dirname(0, 16, keep_head=True) == "coord-0-16"
    assert artifact_subdir(59, 80) == "sub-59-80"
    assert slot_dirname(0, 48, keep_head=True) != slot_dirname(0, 48), "head slice != pure slice"

    # ── resolve_submodel: CACHE hit returns the existing dir without slicing/HF ─
    with tempfile.TemporaryDirectory() as wd:
        d = os.path.join(wd, slot_dirname(16, 48))
        os.makedirs(d)
        open(os.path.join(d, "model.safetensors"), "w").close()   # pretend a prior slice exists
        got = resolve_submodel("Qwen/Qwen2.5-72B-Instruct-AWQ", 16, 48, work_dir=wd, repo=None)
        assert got == d, f"cache hit should return {d}, got {got}"
        # keep_head cache is a SEPARATE dir (won't be satisfied by the pure-stage cache)
        dh = os.path.join(wd, slot_dirname(0, 48, keep_head=True))
        os.makedirs(dh); open(os.path.join(dh, "model.safetensors"), "w").close()
        assert resolve_submodel("m", 0, 48, work_dir=wd, keep_head=True, repo=None) == dh

    # ── parse_layout: first slice keeps head; contiguity enforced ─────────────
    assert pub.parse_layout("0:59,59:80") == [(0, 59, True), (59, 80, False)]
    assert pub.parse_layout("0:16,16:48,48:80") == [(0, 16, True), (16, 48, False), (48, 80, False)]
    for bad in ("0:59,60:80",      # non-contiguous (gap)
                "10:5",            # end <= start
                "",                # empty
                "0:40,40:30"):     # second range inverted
        try:
            pub.parse_layout(bad); raise AssertionError(f"should reject layout {bad!r}")
        except ValueError:
            pass

    # ── build_manifest: structure a node reads to find its slot ───────────────
    m = pub.build_manifest("Qwen/Qwen2.5-72B-Instruct-AWQ", pub.parse_layout("0:59,59:80"))
    assert m["num_layers"] == 80 and m["format"] == "awq-per-node-v1"
    assert m["slots"][0] == {"start": 0, "end": 59, "keep_head": True, "dir": "coord-0-59"}
    assert m["slots"][1] == {"start": 59, "end": 80, "keep_head": False, "dir": "sub-59-80"}
    # slot dirs in the manifest match what a node computes from its assigned range
    for sl in m["slots"]:
        assert sl["dir"] == slot_dirname(sl["start"], sl["end"], sl["keep_head"])

    print("SHARD-FETCH TESTS PASSED — slot naming (coord vs stage), resolve_submodel cache path, "
          "parse_layout contiguity, build_manifest ↔ node slot-name agreement")


if __name__ == "__main__":
    main()
