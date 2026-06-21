"""
test_concurrent.py — pipeline overlap correctness: concurrent requests don't corrupt
each other.

With max_concurrency>1 each in-flight request gets its OWN stage sockets (a pooled
_Conn via request_gate); the framed wire can't interleave and per-request KV stays
isolated. We fire M concurrent generations through 2 real stage workers and require
every one to be token-identical to the sequential reference — proving no socket or KV
cross-talk. (The throughput win itself is a GPU measurement; this is the correctness
gate that must hold before raising concurrency on the live engine.)

Run on CPU:  python3 -m tests.test_concurrent
"""

import os
import socket
import subprocess
import sys
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from transformers.cache_utils import DynamicCache  # noqa: E402

from engine import wire  # noqa: E402
from engine.coordinator import Coordinator  # noqa: E402

MODEL = os.environ.get("CIRCUIT_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
PROMPT = "The capital of France is"
N_NEW = 20
M = 4                     # concurrent requests
CONCURRENCY = 4


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
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
    print(f"launched 2 stage workers; max_concurrency={CONCURRENCY}, firing {M} concurrent requests")

    coord = None
    try:
        coord = Coordinator(MODEL, [("127.0.0.1", p0), ("127.0.0.1", p1)], key,
                            max_concurrency=CONCURRENCY)
        ref = _reference(coord._model, coord.tok, N_NEW)

        results = [None] * M
        errors = [None] * M

        def run(i):
            try:
                with coord.request_gate():
                    _, toks = coord.generate(PROMPT, N_NEW)
                results[i] = toks
            except Exception as e:   # noqa: BLE001
                errors[i] = e

        threads = [threading.Thread(target=run, args=(i,)) for i in range(M)]
        t0 = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        dt = time.time() - t0

        assert not any(errors), f"a concurrent request raised: {errors}"
        oks = [results[i] == ref for i in range(M)]
        print(f"  {sum(oks)}/{M} concurrent requests token-identical to reference  ({dt:.1f}s)")
        for i in range(M):
            assert results[i] == ref, f"request {i} diverged — concurrency corrupted it"

        # and single-stream (max_concurrency=1) still works through the same paths
        single = Coordinator(MODEL, [("127.0.0.1", p0), ("127.0.0.1", p1)], key, max_concurrency=1)
        try:
            _, toks1 = single.generate(PROMPT, N_NEW)
            assert toks1 == ref, "single-stream (concurrency=1) diverged"
        finally:
            single.close()
        print("  single-stream (concurrency=1) path still correct")

        print("CONCURRENT-CORRECTNESS PASSED — overlapping requests stay isolated "
              "(no socket/KV cross-talk); output token-identical to sequential")
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
