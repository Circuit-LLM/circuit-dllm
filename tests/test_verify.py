"""
test_verify.py — the noise-tolerant agreement metric + challenge probe (docs/VERIFICATION.md).

Needs torch (the metric operates on tensors), so it runs on a torch box, not the CPU-only
control-plane host:  python3 -m tests.test_verify

Covers: identical & honest-noise outputs AGREE; garbage / wrong-scale / zero-collapse / shape
mismatch / NaN DISAGREE; challenge_activation is reproducible and correctly shaped.
"""

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch  # noqa: E402

from engine.verify import outputs_agree, challenge_activation  # noqa: E402


def test_identical_and_honest_noise_agree():
    torch.manual_seed(0)
    a = torch.randn(1, 4, 64)
    ok, cos, rel = outputs_agree(a, a.clone())
    assert ok and cos > 0.999 and rel < 1e-5, (ok, cos, rel)

    # honest cross-GPU rounding ≈ tiny relative perturbation → still agrees
    noisy = a + a.abs().mean() * 1e-3 * torch.randn_like(a)
    ok, cos, rel = outputs_agree(noisy, a)
    assert ok, f"honest fp noise must pass (cos={cos:.5f}, rel={rel:.5f})"


def test_garbage_and_wrong_scale_disagree():
    torch.manual_seed(1)
    a = torch.randn(1, 4, 64)
    # unrelated random output → low cosine
    ok, cos, _ = outputs_agree(torch.randn(1, 4, 64), a)
    assert not ok and cos < 0.5, f"garbage must fail (cos={cos:.5f})"
    # right DIRECTION but wrong SCALE (2x) → cosine passes, relative-L2 catches it
    ok, cos, rel = outputs_agree(a * 2.0, a)
    assert not ok and cos > 0.999 and rel > 0.5, f"wrong-scale must fail on L2 (cos={cos}, rel={rel})"


def test_degenerate_inputs_disagree():
    a = torch.randn(1, 4, 64)
    assert not outputs_agree(torch.zeros_like(a), a)[0], "zero-collapse vs real output disagrees"
    assert not outputs_agree(a, torch.full_like(a, float("nan")))[0], "NaN disagrees"
    assert not outputs_agree(a, torch.full_like(a, float("inf")))[0], "Inf disagrees"
    assert not outputs_agree(a, torch.randn(1, 4, 32))[0], "shape mismatch disagrees"
    assert outputs_agree(torch.zeros(1, 4, 8), torch.zeros(1, 4, 8))[0], "both-zero agree"


def test_challenge_is_reproducible_and_shaped():
    x1 = challenge_activation("node-abc:0:nonce", width=128, length=4)
    x2 = challenge_activation("node-abc:0:nonce", width=128, length=4)
    x3 = challenge_activation("node-abc:0:OTHER", width=128, length=4)
    assert tuple(x1.shape) == (1, 4, 128), x1.shape
    assert torch.equal(x1, x2), "same seed → identical challenge (reproducible/auditable)"
    assert not torch.equal(x1, x3), "different nonce → different challenge (unpredictable)"


def main():
    test_identical_and_honest_noise_agree()
    test_garbage_and_wrong_scale_disagree()
    test_degenerate_inputs_disagree()
    test_challenge_is_reproducible_and_shaped()
    print("VERIFY TESTS PASSED — agreement metric (noise vs garbage/scale/degenerate) + challenge probe")


if __name__ == "__main__":
    main()
