"""
test_chain.py — chain-relay routing primitives (pure logic, no GPUs).

Covers: route encode/decode round-trip (with the trailing activation bytes preserved),
multi-hop + empty + single, pop_next, chain_head_and_route from pinned Nodes (incl. the
single-node degrade-to-star case), and malformed-header / bad-input guards.

  python3 -m tests.test_chain
"""

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from engine.chain import encode_route, decode_route, pop_next, chain_head_and_route  # noqa: E402


class _FakeNode:
    def __init__(self, node_id, host, port, key):
        self.node_id = node_id
        self.endpoint = (host, port)
        self.wire_key = key


def main():
    k0 = b"\x00" * 32
    k1 = bytes(range(32))
    k2 = b"\xab" * 32
    act = b"PACKED-ACTIVATION-TENSOR-BYTES"   # stand-in for pack_activation(...) output

    # ── round-trip: route header + trailing activation survive exactly ────────
    hops = [("10.0.0.1", 19210, k1), ("eu-host.example", 443, k2)]
    payload = encode_route(hops) + act
    got, rest = decode_route(payload)
    assert got == hops, "route hops round-trip"
    assert rest == act, "activation bytes preserved after the route header"

    # ── empty + single-hop ────────────────────────────────────────────────────
    assert decode_route(encode_route([]) + act) == ([], act), "empty route"
    one = [("h", 1, k0)]
    assert decode_route(encode_route(one) + b"") == (one, b""), "single hop, no trailer"

    # ── pop_next ──────────────────────────────────────────────────────────────
    nxt, rem = pop_next(hops)
    assert nxt == hops[0] and rem == hops[1:], "pop_next splits head/rest"
    assert pop_next([]) == (None, []), "tail → (None, [])"

    # ── chain_head_and_route from pinned Nodes ────────────────────────────────
    n0 = _FakeNode("n0", "10.0.0.1", 19210, k0)
    n1 = _FakeNode("n1", "10.0.0.2", 19210, k1)
    n2 = _FakeNode("n2", "10.0.0.3", 19210, k2)
    head, downstream = chain_head_and_route([n0, n1, n2])
    assert head is n0, "head is the first slot's holder"
    assert downstream == [("10.0.0.2", 19210, k1), ("10.0.0.3", 19210, k2)], "downstream after head"
    # the coordinator's frame to the head carries `downstream`; re-encode/decode it
    re_hops, _ = decode_route(encode_route(downstream) + act)
    assert re_hops == downstream, "downstream survives the wire"

    # single-node route → (head, []) → degrades to one hop, no chain overhead
    h1, d1 = chain_head_and_route([n0])
    assert h1 is n0 and d1 == [], "single-node route degrades to star"

    # ── guards ────────────────────────────────────────────────────────────────
    for bad in (("h", 70000, k0), ("h", 1, b"short")):
        try:
            encode_route([bad]); assert False, f"should reject {bad}"
        except ValueError:
            pass
    try:
        decode_route(b"\x05")  # claims 5 hops, no body
        assert False, "truncated header should raise"
    except ValueError:
        pass
    # a DOWNSTREAM node needs a 32-byte key (its key is encoded in the route); the HEAD's
    # key is the coordinator's own and is never encoded, so a 1-node route skips the check.
    try:
        chain_head_and_route([n0, _FakeNode("x", "h", 1, None)])
        assert False, "downstream node without wire_key should raise"
    except ValueError:
        pass
    h_ok, d_ok = chain_head_and_route([_FakeNode("solo", "h", 1, None)])
    assert h_ok.node_id == "solo" and d_ok == [], "1-node route OK even with no head key"

    print("CHAIN TESTS PASSED — route round-trip, empty/single, pop_next, "
          "head/downstream, single-node degrade, malformed/bad-input guards")


if __name__ == "__main__":
    main()
