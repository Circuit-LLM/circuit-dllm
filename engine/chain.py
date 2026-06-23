"""
chain.py — chain-relay routing primitives (topology-aware speed; see docs/CHAIN_RELAY.md).

The star pipeline (`coordinator._relay_dynamic`) returns the activation to the coordinator
between EVERY stage → N coordinator round-trips per forward. The CHAIN relay forwards the
activation node→node so the coordinator only touches entry + exit: it sends to the HEAD with
the rest of the route attached, each node computes its layers then forwards to the next hop
(carrying the shortened route), and the final result bubbles back up the chain.

This module is the PURE, GPU-free core: (de)serialize the downstream route carried ahead of
the activation tensor in a CHAIN_ACTIVATION frame, and the per-node "pop my next hop, forward
the rest" step. No torch, no sockets → deterministic unit tests (tests/test_chain.py). The
stage-worker / coordinator wiring builds on these.
"""
from __future__ import annotations

import struct
from typing import List, Optional, Tuple

# A downstream hop the activation should be forwarded to: where to dial + the wire key to
# encrypt to it. (host, port, key[32]). Keys ride in the route only on a permissioned mesh;
# the open network uses per-session forwarding keys instead — see docs/CHAIN_RELAY.md.
Hop = Tuple[str, int, bytes]


def encode_route(hops: List[Hop]) -> bytes:
    """Serialize a downstream route — count, then per hop: host_len, host, port, key. This
    prefixes a CHAIN_ACTIVATION payload (the packed activation tensor follows it)."""
    out = bytearray()
    if len(hops) > 255:
        raise ValueError("route too long (max 255 hops)")
    out.append(len(hops))
    for host, port, key in hops:
        hb = host.encode("utf-8")
        if len(hb) > 255:
            raise ValueError("host too long (max 255 bytes)")
        if not (0 <= port <= 65535):
            raise ValueError(f"port out of range: {port}")
        if len(key) != 32:
            raise ValueError("wire key must be 32 bytes")
        out.append(len(hb))
        out += hb
        out += struct.pack(">H", port)
        out += key
    return bytes(out)


def decode_route(buf: bytes) -> Tuple[List[Hop], bytes]:
    """Inverse of encode_route. Returns (hops, rest), where `rest` is the activation bytes
    that followed the route header. Raises on a truncated/malformed header."""
    if not buf:
        raise ValueError("empty chain payload")
    n = buf[0]
    off = 1
    hops: List[Hop] = []
    for _ in range(n):
        if off >= len(buf):
            raise ValueError("truncated route header")
        hl = buf[off]; off += 1
        host = buf[off:off + hl].decode("utf-8"); off += hl
        (port,) = struct.unpack(">H", buf[off:off + 2]); off += 2
        key = bytes(buf[off:off + 32]); off += 32
        if len(key) != 32:
            raise ValueError("truncated key in route header")
        hops.append((host, port, key))
    return hops, bytes(buf[off:])


def pop_next(hops: List[Hop]) -> Tuple[Optional[Hop], List[Hop]]:
    """Split a downstream route into (next hop or None if tail, remaining route)."""
    if not hops:
        return None, []
    return hops[0], list(hops[1:])


def chain_head_and_route(route) -> Tuple[object, List[Hop]]:
    """From a pinned route (topology Nodes in slot order, each with `.endpoint=(host,port)`
    and `.wire_key`), return (head_node, downstream_hops) — the coordinator sends the
    activation to head_node carrying downstream_hops; each node pops its next + forwards.

    A single-node route yields (node, []) → the head is also the tail (it returns the result
    directly), i.e. chain mode degrades to the star's single hop with no overhead."""
    if not route:
        raise ValueError("empty route")
    head = route[0]
    downstream: List[Hop] = []
    for n in route[1:]:
        host, port = n.endpoint[0], int(n.endpoint[1])
        if n.wire_key is None or len(n.wire_key) != 32:
            raise ValueError(f"node {getattr(n, 'node_id', '?')} has no 32-byte wire_key")
        downstream.append((host, port, n.wire_key))
    return head, downstream
