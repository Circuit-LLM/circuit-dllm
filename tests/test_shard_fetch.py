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

    # ── catalog → coordinator Topology alignment (the download-path guarantee) ─
    from engine.shard_fetch import catalog_layout, topology_from_catalog
    from engine.topology import Topology

    # catalog from a layout string and from a manifest dict agree
    lay = "0:59,59:80"
    assert catalog_layout(lay) == [(0, 59, True), (59, 80, False)]
    assert catalog_layout(m) == [(0, 59, True), (59, 80, False)], "manifest dict → same layout"

    # layout → coordinator Topology args
    assert topology_from_catalog(80, catalog_layout(lay)) == (59, 1, [21])
    assert topology_from_catalog(80, catalog_layout("0:16,16:48,48:80")) == (16, 2, [32, 32])

    # THE GUARANTEE: a Topology built from the catalog assigns exactly the published stage slots,
    # so a node's assigned range maps to a published artifact dir (download, not local-slice).
    ranges = catalog_layout("0:16,16:48,48:80")
    ce, ns, sizes = topology_from_catalog(80, ranges)
    topo = Topology(num_layers=80, coordinator_end=ce, num_stages=ns, model_fp="m", slot_sizes=sizes)
    assigned = [(s.start, s.end) for s in topo.slots]
    published_stage_dirs = [slot_dirname(s, e, False) for (s, e, kh) in ranges if not kh]
    node_resolved_dirs = [slot_dirname(s.start, s.end, False) for s in topo.slots]
    assert assigned == [(16, 48), (48, 80)], assigned
    assert node_resolved_dirs == published_stage_dirs, "coordinator slots ↔ published artifact dirs"

    # validation: first slot must be keep-head; stages must tile to num_layers
    for bad in (lambda: topology_from_catalog(80, [(0, 59, False), (59, 80, False)]),  # no head
                lambda: topology_from_catalog(80, [(0, 59, True)]),                    # no stages
                lambda: topology_from_catalog(80, [(0, 40, True), (40, 70, False)])):  # gap to 80
        try:
            bad(); raise AssertionError("should reject bad catalog")
        except ValueError:
            pass

    # ── integrity + revision pinning (best-practice hardening) ────────────────
    from engine.shard_fetch import sha256_file, verify_slice, build_manifest as bmf
    with tempfile.TemporaryDirectory() as wd:
        d = os.path.join(wd, "sub-16-48"); os.makedirs(d)
        w = os.path.join(d, "model.safetensors")
        with open(w, "wb") as f:
            f.write(b"hello-awq-slice")
        import hashlib
        want = hashlib.sha256(b"hello-awq-slice").hexdigest()
        assert sha256_file(w) == want, "streaming sha256 matches hashlib"
        assert verify_slice(d, want) is True, "matching hash verifies"
        assert verify_slice(d, "deadbeef") is False, "wrong hash rejected"
        assert verify_slice(d, "") is True and verify_slice(d, None) is True, "no expected hash → skip"

    # manifest carries revision + per-slot sha256/bytes
    mh = bmf("Qwen/Qwen2.5-72B-Instruct-AWQ", pub.parse_layout("0:59,59:80"),
             revision="abc123", slot_meta={"sub-59-80": {"sha256": "ff00", "bytes": 42}})
    assert mh["revision"] == "abc123", "manifest pins source revision"
    by_dir = {s["dir"]: s for s in mh["slots"]}
    assert by_dir["sub-59-80"]["sha256"] == "ff00" and by_dir["sub-59-80"]["bytes"] == 42
    assert "sha256" not in by_dir["coord-0-59"], "slots without meta carry no hash (not yet hashed)"
    # back-compat: build_manifest still works with no revision/meta (old callers)
    assert bmf("m", pub.parse_layout("0:16,16:48,48:80"))["revision"] is None

    print("SHARD-FETCH TESTS PASSED — slot naming, resolve_submodel cache, parse_layout, "
          "build_manifest↔node agreement, catalog→Topology alignment, sha256/verify_slice + "
          "revision-pinned manifest (integrity hardening)")


if __name__ == "__main__":
    main()
