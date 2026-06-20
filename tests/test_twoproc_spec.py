"""
test_twoproc_spec.py — speculative decode over the real two-process socket.

Launches two stage workers and runs coordinator.generate_speculative (local
draft proposes, the split stages verify over the wire, KV rolls back via
KV_CTRL). Output must equal an in-process unsplit greedy reference.

Run on RunPod:  python3 -m tests.test_twoproc_spec
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
N_NEW = 24
K = 4


def free_port() -> int:
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close()
    return p


def _reference(model, tok, n_new):
    ids = tok(PROMPT, return_tensors="pt").input_ids
    cache = DynamicCache(config=model.config)
    nxt = model(ids, past_key_values=cache, use_cache=True).logits[:, -1].argmax(-1, keepdim=True)
    out = [int(nxt)]
    for _ in range(n_new - 1):
        nxt = model(nxt, past_key_values=cache, use_cache=True).logits[:, -1].argmax(-1, keepdim=True)
        out.append(int(nxt))
    return out


def main():
    key = os.urandom(wire.KEY_LEN)
    keyhex = key.hex()
    p0, p1 = free_port(), free_port()
    procs = [subprocess.Popen(
        [sys.executable, "-m", "engine.stage_worker", "--port", str(port),
         "--layers", layers, "--model", MODEL, "--key", keyhex, "--device", "cpu"],
        cwd=REPO) for port, layers in [(p0, "0:12"), (p1, "12:24")]]
    print(f"launched 2 stage workers on {p0}, {p1}")

    coord = None
    try:
        coord = Coordinator(MODEL, [("127.0.0.1", p0), ("127.0.0.1", p1)], key)
        t0 = time.time()
        text, got = coord.generate_speculative(PROMPT, N_NEW, K=K)
        dt = time.time() - t0
        ref = _reference(coord._model, coord.tok, N_NEW)
        match = got == ref
        print(f"  socket speculative decode: {match}  ({N_NEW} tokens, {dt:.1f}s)")
        print(f"  output: {text!r}")
        if not match:
            print("  ref:", ref)
            print("  got:", got)
        assert match, "socket speculative decode diverged from greedy reference"
        print("TWO-PROCESS SPECULATIVE DECODE PASSED")
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
