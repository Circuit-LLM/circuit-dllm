"""
test_tensors.py — tensor serialization round-trip (byte-exact, incl. bfloat16).

Needs torch. Run from circuit-engine/:  python -m tests.test_tensors
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from engine import tensors as T  # noqa: E402


def test_roundtrip_dtypes():
    for dt in (torch.float32, torch.float16, torch.bfloat16, torch.int32, torch.int64, torch.uint8):
        if dt.is_floating_point:
            x = torch.randn(3, 17, 5).to(dt)
        else:
            x = torch.randint(0, 100, (3, 17, 5), dtype=dt)
        y = T.unpack_tensor(T.pack_tensor(x))
        assert y.dtype == dt, f"dtype drift {dt} -> {y.dtype}"
        assert y.shape == x.shape, f"shape drift for {dt}"
        # byte-exact equality (bitwise via uint8 view, so NaN/precision can't lie)
        assert torch.equal(x.view(torch.uint8), y.view(torch.uint8)), f"bytes differ for {dt}"
    print("  ok: round-trip byte-exact for 6 dtypes (incl. bfloat16)")


def test_noncontiguous():
    x = torch.randn(8, 4, dtype=torch.float32).t()  # non-contiguous view
    assert not x.is_contiguous()
    y = T.unpack_tensor(T.pack_tensor(x))
    assert torch.equal(x.contiguous().view(torch.uint8), y.view(torch.uint8))
    print("  ok: non-contiguous tensor packs correctly")


def test_activation_payload():
    h = torch.randn(1, 12, 64, dtype=torch.bfloat16)
    payload = T.pack_activation(session_id=4242, position=7, hidden=h, flags=T.FLAG_LAST_STAGE)
    sid, pos, flags, h2 = T.unpack_activation(payload)
    assert sid == 4242 and pos == 7 and flags == T.FLAG_LAST_STAGE
    assert torch.equal(h.view(torch.uint8), h2.view(torch.uint8))
    print("  ok: activation payload (session/pos/flags + bf16 hidden) round-trips")


def test_large_hidden():
    # a realistic single-token hidden state for a 32B (hidden=5120), fp16
    h = torch.randn(1, 1, 5120, dtype=torch.float16)
    sid, pos, flags, h2 = T.unpack_activation(T.pack_activation(1, 0, h))
    assert torch.equal(h.view(torch.uint8), h2.view(torch.uint8))
    print("  ok: 5120-dim fp16 hidden state round-trips")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"tensors.py self-test — {len(tests)} cases")
    for t in tests:
        t()
    print("ALL TENSOR TESTS PASSED")


if __name__ == "__main__":
    main()
