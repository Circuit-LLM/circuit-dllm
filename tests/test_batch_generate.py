"""
test_batch_generate.py — Win B / B3a: Coordinator.generate_batch token-identical to
per-sequence generate, in the LIVE shape (co-located local stage + one remote stage).

This is the batched-generation primitive the scheduler will drive: a fixed batch of
ragged-length prompts decoded together (one batched forward per step, through the
coordinator's co-located stage 0:12 AND a remote stage 12:24), each sequence stopping
at EOS. Every row must equal what generate() produces for that prompt alone.

Run on CPU:  python3 -m tests.test_batch_generate
"""

import os
import socket
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from engine import wire  # noqa: E402
from engine.coordinator import Coordinator  # noqa: E402

MODEL = os.environ.get("CIRCUIT_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
N_NEW = 24
PROMPTS = [
    "The capital of France is",
    "Two plus two equals",
    "The largest planet in the solar system is the gas giant",
    "Water",
]


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def wait_listen(port, timeout=90):
    end = time.time() + timeout
    while time.time() < end:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(0.5)
    raise RuntimeError(f"port {port} never came up")


def main():
    key = os.urandom(wire.KEY_LEN)
    port = free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "engine.stage_worker", "--port", str(port),
         "--layers", "12:24", "--model", MODEL, "--key", key.hex(), "--device", "cpu"],
        cwd=REPO)
    print("co-located stage 0:12 + remote stage 12:24 (the live shape)")

    coord = None
    try:
        wait_listen(port)
        coord = Coordinator(MODEL, [("127.0.0.1", port)], key, local_layers=(0, 12), device="cpu")

        # per-sequence reference through the SAME coordinator (trusted single-seq path)
        refs = [coord.generate(p, N_NEW)[1] for p in PROMPTS]

        t0 = time.time()
        batched = coord.generate_batch(PROMPTS, N_NEW)
        dt = time.time() - t0

        allok = True
        for b, p in enumerate(PROMPTS):
            ok = batched[b] == refs[b]
            allok = allok and ok
            print(f"  [{'OK ' if ok else 'BAD'}] seq {b} ({len(batched[b])} tok): {coord.tok.decode(batched[b])!r}")
            if not ok:
                print(f"        ref: {coord.tok.decode(refs[b])!r}")
                print(f"        got: {batched[b]}")
                print(f"        ref: {refs[b]}")
        assert allok, "generate_batch diverged from per-sequence generate"
        print(f"B3a PASSED — generate_batch token-identical to per-seq generate, "
              f"co-located + remote stage ({dt:.1f}s for {len(PROMPTS)} seqs)")
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
