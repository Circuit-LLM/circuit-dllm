"""
verify.py — trustless node verification primitives (docs/VERIFICATION.md).

A stage's output for a given input is deterministic only up to floating-point rounding, which
*differs* across GPUs/kernels (the mesh is numerically non-deterministic — the same prompt at
temp 0 can yield different tokens run to run). So two HONEST holders of the same layer slice
agree to high precision but NOT bit-exactly, while a broken / lazy / malicious node's output
diverges far more. `outputs_agree` draws that line with a noise-tolerant metric;
`challenge_activation` makes a reproducible probe input to feed both nodes.

Pure + torch only — unit-testable on CPU, no mesh.
"""

from __future__ import annotations

import hashlib

import torch


def _seed_from(text: str) -> int:
    """Stable 63-bit seed from a string (node_id + nonce), so a challenge is reproducible for
    replay/audit yet unpredictable enough that a node can't precompute a lookup table."""
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big") & 0x7FFFFFFFFFFFFFFF


def challenge_activation(seed: str, width: int, length: int = 4,
                         dtype: torch.dtype = torch.float16,
                         device: str = "cpu") -> torch.Tensor:
    """A reproducible pseudo-random hidden state [1, length, width] for a challenge. Generated on
    CPU from `seed` (deterministic across machines), then moved to `device`/`dtype`. Width = the
    model's residual-stream size (config.hidden_size); both the node-under-test and the trusted
    reference run their identical layer slice on it."""
    g = torch.Generator(device="cpu").manual_seed(_seed_from(seed))
    x = torch.randn(1, length, width, generator=g, dtype=torch.float32)
    return x.to(dtype=dtype, device=device)


def outputs_agree(a, b, cos_thresh: float = 0.99, rel_l2_thresh: float = 0.05):
    """Do two stage outputs for the SAME input match within numerical noise?

    Requires BOTH high cosine similarity (direction) AND low relative L2 (magnitude): cosine
    alone passes a correct-direction / wrong-scale output, L2 alone is noisy at large magnitudes.
    Shape mismatch, NaN, or Inf → disagree. Returns (ok, cos, rel_l2) so the auditor can log the
    margin. Defaults are conservative starting points — tune against real cross-GPU spread.
    """
    if a is None or b is None or tuple(a.shape) != tuple(b.shape):
        return False, 0.0, float("inf")
    a = a.detach().to(torch.float32).flatten()
    b = b.detach().to(torch.float32).flatten()
    if not (bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all())):
        return False, 0.0, float("inf")
    na, nb = float(a.norm()), float(b.norm())
    if na == 0.0 and nb == 0.0:
        return True, 1.0, 0.0          # both genuinely zero → agree
    if na == 0.0 or nb == 0.0:
        return False, 0.0, float("inf")  # one side collapsed to zero → disagree
    cos = float(torch.dot(a, b) / (na * nb))
    rel_l2 = float((a - b).norm() / nb)
    ok = (cos >= cos_thresh) and (rel_l2 <= rel_l2_thresh)
    return ok, cos, rel_l2
