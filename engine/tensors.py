"""
tensors.py — compact tensor <-> bytes serialization for pipeline activations.

Original implementation. An ACTIVATION frame carries a small header (session id,
sequence position, a flags byte) plus a hidden-state tensor. We serialize the
tensor as [dtype, ndim, shape...] + raw contiguous bytes, reconstructing with
torch.frombuffer so it stays correct for bfloat16 (which numpy can't represent).

Wire layout of a packed tensor:

    [dtype_code u8][ndim u8][dim0 u32 BE]...[dimN u32 BE][ raw contiguous bytes ]

Activation payload (what rides inside a wire.ACTIVATION frame):

    [session_id u32 BE][position u32 BE][flags u8][ packed tensor ]
"""

from __future__ import annotations

import struct
from typing import Tuple

import torch

# dtype registry — stable codes; append, never renumber.
_CODE_TO_DT = {
    1: torch.float32,
    2: torch.float16,
    3: torch.bfloat16,
    4: torch.int32,
    5: torch.int64,
    6: torch.uint8,
}
_DT_TO_CODE = {v: k for k, v in _CODE_TO_DT.items()}

# flags
FLAG_NONE = 0
FLAG_LAST_STAGE = 1 << 0   # this activation is the final stage's output


class TensorError(Exception):
    pass


def pack_tensor(t: torch.Tensor) -> bytes:
    """Serialize a tensor to bytes. Moves to CPU + contiguous; preserves dtype/shape."""
    if t.dtype not in _DT_TO_CODE:
        raise TensorError(f"unsupported dtype {t.dtype}")
    if t.dim() > 255:
        raise TensorError("too many dimensions")
    t = t.detach().to("cpu").contiguous()
    head = bytes([_DT_TO_CODE[t.dtype], t.dim()]) + b"".join(
        struct.pack(">I", d) for d in t.shape
    )
    raw = t.view(torch.uint8).numpy().tobytes()  # byte-reinterpret: bf16-safe
    return head + raw


def unpack_tensor(buf: bytes) -> torch.Tensor:
    """Inverse of pack_tensor."""
    if len(buf) < 2:
        raise TensorError("buffer too short")
    code, ndim = buf[0], buf[1]
    if code not in _CODE_TO_DT:
        raise TensorError(f"unknown dtype code {code}")
    dt = _CODE_TO_DT[code]
    off = 2
    shape = []
    for _ in range(ndim):
        (d,) = struct.unpack(">I", buf[off:off + 4])
        shape.append(d)
        off += 4
    raw = buf[off:]
    flat = torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(dt)
    numel = 1
    for d in shape:
        numel *= d
    if flat.numel() != numel:
        raise TensorError(f"byte count mismatch: have {flat.numel()} elems, shape wants {numel}")
    return flat.reshape(shape).clone()  # clone: own the memory, detach from buffer


def pack_activation(session_id: int, position: int, hidden: torch.Tensor,
                    flags: int = FLAG_NONE) -> bytes:
    """Build the payload for a wire.ACTIVATION frame."""
    return struct.pack(">IIB", session_id & 0xFFFFFFFF, position & 0xFFFFFFFF, flags) \
        + pack_tensor(hidden)


def unpack_activation(payload: bytes) -> Tuple[int, int, int, torch.Tensor]:
    """Inverse of pack_activation -> (session_id, position, flags, hidden)."""
    if len(payload) < 9:
        raise TensorError("activation payload too short")
    session_id, position, flags = struct.unpack(">IIB", payload[:9])
    hidden = unpack_tensor(payload[9:])
    return session_id, position, flags, hidden


def pack_batch_activation(batch_id: int, position: int, hidden: torch.Tensor,
                          position_ids: torch.Tensor, attention_mask: torch.Tensor,
                          flags: int = FLAG_NONE) -> bytes:
    """Payload for a batched activation (Win B): the [B,T,D] hidden plus the per-row
    position_ids [B,T] and the 2D padding mask [B,kv_len] the stage needs to attend
    each row to its own ragged KV. Length-delimited so the three tensors unpack cleanly."""
    parts = [pack_tensor(hidden),
             pack_tensor(position_ids.to(torch.int32)),
             pack_tensor(attention_mask.to(torch.uint8))]
    body = b"".join(struct.pack(">I", len(p)) + p for p in parts)
    return struct.pack(">IIB", batch_id & 0xFFFFFFFF, position & 0xFFFFFFFF, flags) + body


def unpack_batch_activation(payload: bytes):
    """Inverse -> (batch_id, position, flags, hidden, position_ids, attention_mask)."""
    if len(payload) < 9:
        raise TensorError("batch activation payload too short")
    batch_id, position, flags = struct.unpack(">IIB", payload[:9])
    off = 9
    tensors = []
    for _ in range(3):
        (n,) = struct.unpack(">I", payload[off:off + 4])
        off += 4
        tensors.append(unpack_tensor(payload[off:off + n]))
        off += n
    hidden, position_ids, attention_mask = tensors
    return batch_id, position, flags, hidden, position_ids, attention_mask


__all__ = [
    "FLAG_NONE", "FLAG_LAST_STAGE", "TensorError",
    "pack_tensor", "unpack_tensor", "pack_activation", "unpack_activation",
    "pack_batch_activation", "unpack_batch_activation",
]
