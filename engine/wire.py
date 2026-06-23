"""
wire.py — encrypted, length-prefixed message framing for the Circuit pipeline.

Original implementation. Designed for running stage workers and the coordinator
over the open internet, so every frame is authenticated + encrypted and the
reader is bounded against oversized/garbage input.

Frame on the wire (big-endian length prefix so a reader can find boundaries):

    +----------------+------------------+--------------------------------+
    | length  (4 B)  | nonce   (12 B)   | ciphertext + tag (length-12 B) |
    +----------------+------------------+--------------------------------+

    length      = len(nonce) + len(ciphertext+tag)
    ciphertext  = ChaCha20-Poly1305(key).encrypt(nonce, plaintext, aad=VERSION)
    plaintext   = msg_type (1 B) || payload (N B)

The 1-byte protocol VERSION is passed as AEAD associated data, so it is
authenticated (a version mismatch fails the tag check) without being secret.
The key is the 32-byte pre-shared cluster key. Nonces are random per frame
(96-bit random nonces are safe for ChaCha20-Poly1305 well within a session's
frame budget).

This module is pure bytes + crypto: no torch, no sockets-specific assumptions
beyond a blocking recv/sendall interface. Tensor (de)serialization lives in
tensors.py so this stays GPU-free and unit-testable on any host.
"""

from __future__ import annotations

import os
import struct
from typing import Tuple

from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.exceptions import InvalidTag

# --- protocol constants ----------------------------------------------------

VERSION = 1
_VERSION_AAD = bytes([VERSION])

NONCE_LEN = 12
LEN_PREFIX = 4
TAG_LEN = 16
KEY_LEN = 32

# Cap a single frame. Activations are small, but chunked weight shards can be
# large; 256 MiB is a generous ceiling that still refuses a memory-DoS frame.
MAX_FRAME = 256 * 1024 * 1024

# Message types (1 byte). Keep stable; append new ones, never renumber.
HELLO = 1       # worker -> coordinator: identity + capabilities
WELCOME = 2     # coordinator -> worker: assignment (layer range, model cfg)
ACTIVATION = 3  # hidden states flowing forward through the pipeline
RESULT = 4      # stage output / final logits|argmaxes returning
KV_CTRL = 5     # session control: accept M / rollback / reset
PING = 6
PONG = 7
ERROR = 8
BYE = 9
BATCH_ACTIVATION = 10  # batched hidden states (Win B): [B,T,D] + per-row pos + 2D mask
CHAIN_ACTIVATION = 11  # chain relay: encode_route(downstream) ++ pack_activation(...) —
                       # the node forwards node->next instead of returning to the coordinator
                       # (engine/chain.py, docs/CHAIN_RELAY.md). Gated by CIRCUIT_CHAIN.

_NAMES = {
    HELLO: "HELLO", WELCOME: "WELCOME", ACTIVATION: "ACTIVATION", RESULT: "RESULT",
    KV_CTRL: "KV_CTRL", PING: "PING", PONG: "PONG", ERROR: "ERROR", BYE: "BYE",
    BATCH_ACTIVATION: "BATCH_ACTIVATION", CHAIN_ACTIVATION: "CHAIN_ACTIVATION",
}


def msg_name(t: int) -> str:
    return _NAMES.get(t, f"UNKNOWN({t})")


class WireError(Exception):
    """Framing or authentication failure. Always fatal for the connection."""


# --- key handling ----------------------------------------------------------

def normalize_key(key) -> bytes:
    """Accept a 32-byte key, or a 64-char hex string, and return raw 32 bytes."""
    if isinstance(key, str):
        key = bytes.fromhex(key)
    if not isinstance(key, (bytes, bytearray)) or len(key) != KEY_LEN:
        raise WireError(f"cluster key must be {KEY_LEN} bytes (got {len(key)})")
    return bytes(key)


# --- frame encode / decode (pure bytes) ------------------------------------

def seal(key: bytes, msg_type: int, payload: bytes = b"") -> bytes:
    """Encrypt one message and return the complete length-prefixed wire frame."""
    if not (0 <= msg_type <= 255):
        raise WireError("msg_type out of range")
    aead = ChaCha20Poly1305(key)
    nonce = os.urandom(NONCE_LEN)
    plaintext = bytes([msg_type]) + payload
    ct = aead.encrypt(nonce, plaintext, _VERSION_AAD)
    body = nonce + ct
    if len(body) > MAX_FRAME:
        raise WireError(f"frame too large: {len(body)} > {MAX_FRAME}")
    return struct.pack(">I", len(body)) + body


def open_body(key: bytes, body: bytes) -> Tuple[int, bytes]:
    """Decrypt a frame body (nonce + ciphertext, no length prefix)."""
    if len(body) < NONCE_LEN + TAG_LEN + 1:
        raise WireError("frame body too short")
    nonce, ct = body[:NONCE_LEN], body[NONCE_LEN:]
    aead = ChaCha20Poly1305(key)
    try:
        plaintext = aead.decrypt(nonce, ct, _VERSION_AAD)
    except InvalidTag:
        raise WireError("authentication failed (bad key, tampering, or version)")
    return plaintext[0], plaintext[1:]


# --- blocking socket helpers ----------------------------------------------

def recv_exact(sock, n: int) -> bytes:
    """Read exactly n bytes from a blocking socket, or raise on early close."""
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        r = sock.recv_into(view[got:], n - got)
        if r == 0:
            raise WireError("connection closed mid-frame")
        got += r
    return bytes(buf)


def read_frame(sock) -> Tuple[int, bytes]:
    """Read and decrypt one frame from a blocking socket. Needs the key set via
    a bound reader; for the bare helper, use read_frame_keyed."""
    raise WireError("use read_frame_keyed(sock, key)")


def read_frame_keyed(sock, key: bytes) -> Tuple[int, bytes]:
    """Read one length-prefixed frame from a blocking socket and decrypt it."""
    (length,) = struct.unpack(">I", recv_exact(sock, LEN_PREFIX))
    if length > MAX_FRAME or length < NONCE_LEN + TAG_LEN + 1:
        raise WireError(f"invalid frame length {length}")
    body = recv_exact(sock, length)
    return open_body(key, body)


def write_frame(sock, key: bytes, msg_type: int, payload: bytes = b"") -> None:
    """Seal and send one frame on a blocking socket."""
    sock.sendall(seal(key, msg_type, payload))


__all__ = [
    "VERSION", "MAX_FRAME", "KEY_LEN",
    "HELLO", "WELCOME", "ACTIVATION", "RESULT", "KV_CTRL", "PING", "PONG", "ERROR", "BYE",
    "msg_name", "WireError", "normalize_key",
    "seal", "open_body", "recv_exact", "read_frame_keyed", "write_frame",
]
