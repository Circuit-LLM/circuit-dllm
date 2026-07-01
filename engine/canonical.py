"""Canonical JSON for ed25519 mesh signing — the ONE Python serializer.

Byte-identical with the JS peers' `stableStringify` (circuit-sdk `@circuit/core`): sorted keys, compact
separators, and **`ensure_ascii=False`** so non-ASCII stays literal (Python's default escapes it, which
would silently break cross-language signature verification). Python has no `undefined`, so there is
nothing to drop.

Cross-repo contract — see circuit-sdk/docs/canonical-serialization.md. Pinned by
tests/test_canonical_conformance.py against the shared golden vectors.
"""
import json


def canonical_str(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical(obj) -> bytes:
    """The exact bytes signed/verified (canonical JSON, UTF-8)."""
    return canonical_str(obj).encode()
