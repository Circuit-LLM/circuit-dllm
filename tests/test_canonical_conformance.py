# tests/test_canonical_conformance.py — pins circuit-dllm's canonical JSON (engine.canonical, used for
# ed25519 mesh signing in control_server.py + stage_worker.py) to the SHARED golden vectors. Source of
# truth: circuit-sdk, docs/canonical-serialization.md. `ensure_ascii=False` is required so non-ASCII
# matches the JS peers. Copy canonical-vectors.json alongside on update.
import json
import os

from engine.canonical import canonical_str


def _load():
    here = os.path.dirname(__file__)
    with open(os.path.join(here, "canonical-vectors.json"), encoding="utf-8") as f:
        return json.load(f)


def test_canonical_matches_shared_vectors():
    spec = _load()
    # Python has no `undefined`; the undefined-containing vectors can't be constructed here, so we assert
    # the undefined-free subset — which is every real payload (all producers normalize; see the sweep in
    # docs/canonical-serialization.md).
    checked = 0
    for vec in spec["vectors"]:
        if vec["undefinedInput"]:
            continue
        got = canonical_str(vec["input"])
        assert got == vec["canonical"], f'{vec["note"]}: want {vec["canonical"]!r} got {got!r}'
        checked += 1
    assert checked > 0
