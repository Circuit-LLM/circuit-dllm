"""
test_twoproc.py — end-to-end two-process integration.

Launches two real stage-worker processes (layers 0:12 and 12:24) and a
coordinator that talks to them over the encrypted socket, then asserts the
generated tokens are identical to an in-process unsplit greedy reference.

This proves the socket + process path — not just the in-process math.
Run on RunPod:  python3 -m tests.test_twoproc
"""

import os
import socket
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch  # noqa: E402
from transformers.cache_utils import DynamicCache  # noqa: E402

from engine import wire  # noqa: E402
from engine.coordinator import Coordinator  # noqa: E402

MODEL = os.environ.get("CIRCUIT_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
PROMPT = "The capital of France is"
N_NEW = 20


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _reference(model, tok, n_new):
    ids = tok(PROMPT, return_tensors="pt").input_ids
    cache = DynamicCache(config=model.config)
    logits = model(ids, past_key_values=cache, use_cache=True).logits
    nxt = logits[:, -1].argmax(-1, keepdim=True)
    out = [int(nxt)]
    for _ in range(n_new - 1):
        logits = model(nxt, past_key_values=cache, use_cache=True).logits
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        out.append(int(nxt))
    return out


def main():
    key = os.urandom(wire.KEY_LEN)
    keyhex = key.hex()
    p0, p1 = free_port(), free_port()

    procs = []
    for port, layers in [(p0, "0:12"), (p1, "12:24")]:
        procs.append(subprocess.Popen(
            [sys.executable, "-m", "engine.stage_worker", "--port", str(port),
             "--layers", layers, "--model", MODEL, "--key", keyhex, "--device", "cpu"],
            cwd=REPO))
    print(f"launched 2 stage workers on ports {p0}, {p1}")

    coord = None
    try:
        coord = Coordinator(MODEL, [("127.0.0.1", p0), ("127.0.0.1", p1)], key)
        t0 = time.time()
        text, got = coord.generate(PROMPT, N_NEW)
        dt = time.time() - t0
        ref = _reference(coord._model, coord.tok, N_NEW)

        match = got == ref
        print(f"  two-process decode: {match}  ({N_NEW} tokens, {dt:.1f}s)")
        print(f"  output: {text!r}")
        if not match:
            print("  ref:", ref)
            print("  got:", got)
        assert match, "two-process socket decode diverged from in-process reference"
        print("TWO-PROCESS INTEGRATION PASSED")
    finally:
        if coord is not None:
            coord.close()
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()


if __name__ == "__main__":
    main()
