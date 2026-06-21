"""
test_session_cleanup.py — the coordinator must free a session's KV after every
generation, so _local_caches (and, via KV_CTRL RESET, the stage workers' per-
session caches) don't grow unbounded.

Regression test for the KV-cache leak: _reset_sessions existed but had no callers,
so every request permanently retained its KV in coordinator VRAM (eventual OOM).
This mirrors the live topology — a co-located stage 0 (layers 0:12 + head) plus a
remote stage (12:24) — so _local_caches is actually exercised, then asserts it is
empty after the non-stream path, the stream path, and an early-closed stream
(client-disconnect / GeneratorExit).

Run on CPU with the small model:  python3 -m tests.test_session_cleanup
"""

import os
import socket
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from engine import wire  # noqa: E402
from engine.coordinator import Coordinator  # noqa: E402

MODEL = os.environ.get("CIRCUIT_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
PROMPT = "The capital of France is"


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def main():
    key = os.urandom(wire.KEY_LEN)
    p1 = free_port()
    # one remote stage holds layers 12:24; the coordinator co-locates 0:12 + head
    proc = subprocess.Popen(
        [sys.executable, "-m", "engine.stage_worker", "--port", str(p1),
         "--layers", "12:24", "--model", MODEL, "--key", key.hex(), "--device", "cpu"],
        cwd=REPO)
    coord = None
    try:
        coord = Coordinator(MODEL, [("127.0.0.1", p1)], key, device="cpu",
                            local_layers=(0, 12))

        # non-stream: the co-located cache must be emptied after each call, and
        # greedy output must stay deterministic (proves the finally didn't corrupt
        # decode state).
        prev = None
        for i in range(3):
            _, out = coord.generate(PROMPT, 8)
            assert out, "no tokens produced"
            assert len(coord._local_caches) == 0, \
                f"LEAK after generate() #{i}: {len(coord._local_caches)} cache(s) retained"
            if prev is not None:
                assert out == prev, "greedy output not deterministic across runs"
            prev = out

        # stream path: empty after full consumption
        chunks = list(coord.generate_stream(PROMPT, 8))
        assert chunks, "stream produced nothing"
        assert len(coord._local_caches) == 0, \
            f"LEAK after generate_stream(): {len(coord._local_caches)} retained"

        # stream path: empty even when the consumer stops early (client disconnect
        # -> GeneratorExit -> the finally must still run)
        gen = coord.generate_stream(PROMPT, 64)
        next(gen)
        gen.close()
        assert len(coord._local_caches) == 0, \
            f"LEAK after early-closed stream: {len(coord._local_caches)} retained"

        print("SESSION-CLEANUP TEST PASSED — _local_caches empty after every path; greedy deterministic")
    finally:
        if coord is not None:
            coord.close()
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
