"""
test_wire.py — wire protocol round-trip, tamper detection, and socket framing.

Pure CPU, no torch. Run: python3 -m tests.test_wire  (from circuit-engine/)
"""

import os
import socket
import struct
import sys
import threading

# allow running as `python3 tests/test_wire.py` or `-m tests.test_wire`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import wire  # noqa: E402


def _key():
    return os.urandom(wire.KEY_LEN)


def test_roundtrip():
    key = _key()
    for mt, pl in [(wire.HELLO, b""), (wire.ACTIVATION, os.urandom(4096)),
                   (wire.RESULT, b"\x00\x01\x02"), (wire.PING, b"ping")]:
        frame = wire.seal(key, mt, pl)
        # strip the 4-byte length prefix to feed open_body
        (length,) = struct.unpack(">I", frame[:4])
        assert length == len(frame) - 4
        t, p = wire.open_body(key, frame[4:])
        assert t == mt and p == pl, f"roundtrip failed for {wire.msg_name(mt)}"
    print("  ok: roundtrip (4 message types, incl. 4KB payload)")


def test_hex_key_accepted():
    raw = os.urandom(wire.KEY_LEN)
    hexk = raw.hex()
    frame = wire.seal(wire.normalize_key(hexk), wire.ACTIVATION, b"abc")
    t, p = wire.open_body(wire.normalize_key(raw), frame[4:])
    assert t == wire.ACTIVATION and p == b"abc"
    print("  ok: 64-char hex key == raw 32-byte key")


def test_wrong_key_rejected():
    frame = wire.seal(_key(), wire.ACTIVATION, b"secret")
    try:
        wire.open_body(_key(), frame[4:])
    except wire.WireError as e:
        assert "authentication failed" in str(e)
        print("  ok: wrong key -> auth failure (not silent)")
        return
    raise AssertionError("wrong key was NOT rejected")


def test_tamper_rejected():
    key = _key()
    frame = bytearray(wire.seal(key, wire.ACTIVATION, b"0123456789"))
    frame[-1] ^= 0x01  # flip a ciphertext/tag bit
    try:
        wire.open_body(key, bytes(frame[4:]))
    except wire.WireError:
        print("  ok: single-bit tamper -> rejected")
        return
    raise AssertionError("tampered frame was NOT rejected")


def test_version_mismatch_rejected():
    key = _key()
    frame = wire.seal(key, wire.PING, b"x")
    saved = wire._VERSION_AAD
    try:
        wire._VERSION_AAD = bytes([wire.VERSION + 1])  # simulate a peer on v2
        try:
            wire.open_body(key, frame[4:])
        except wire.WireError:
            print("  ok: version mismatch (AAD) -> rejected")
            return
        raise AssertionError("version mismatch was NOT rejected")
    finally:
        wire._VERSION_AAD = saved


def test_oversize_rejected():
    key = _key()
    a, b = socket.socketpair()
    try:
        # send a bogus length prefix claiming > MAX_FRAME
        a.sendall(struct.pack(">I", wire.MAX_FRAME + 1))
        try:
            wire.read_frame_keyed(b, key)
        except wire.WireError as e:
            assert "invalid frame length" in str(e)
            print("  ok: oversize length prefix -> rejected before alloc")
            return
        raise AssertionError("oversize frame was NOT rejected")
    finally:
        a.close(); b.close()


def test_socket_framing():
    """Real socketpair: writer sends 3 frames back-to-back, reader splits them."""
    key = _key()
    a, b = socket.socketpair()
    msgs = [(wire.HELLO, b"node-1"), (wire.ACTIVATION, os.urandom(20000)), (wire.BYE, b"")]

    def writer():
        for mt, pl in msgs:
            wire.write_frame(a, key, mt, pl)
        a.shutdown(socket.SHUT_WR)

    th = threading.Thread(target=writer)
    th.start()
    got = []
    try:
        for _ in msgs:
            got.append(wire.read_frame_keyed(b, key))
    finally:
        th.join(); a.close(); b.close()
    assert got == msgs, "framed messages did not survive the socket round-trip"
    print("  ok: 3 back-to-back frames over a socket (incl. 20KB) split correctly")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"wire.py self-test — {len(tests)} cases")
    for t in tests:
        t()
    print("ALL WIRE TESTS PASSED")


if __name__ == "__main__":
    main()
