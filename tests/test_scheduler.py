"""
test_scheduler.py — Win B / B3: the batching scheduler end to end.

Requests submitted to the BatchScheduler get collected into one batch, decoded
together (one batched forward per step), and streamed back per-request. Every
request's streamed tokens must equal what generate() produces for it alone — proving
the scheduler + streaming preserve the per-row correctness from B1/B2/B3a.

Run on CPU:  python3 -m tests.test_scheduler
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
from engine.scheduler import BatchScheduler  # noqa: E402

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


def drain(out_q):
    toks = []
    while True:
        item = out_q.get(timeout=180)
        if item is BatchScheduler.DONE:
            return toks
        if isinstance(item, Exception):
            raise item
        toks.append(item)


def main():
    key = os.urandom(wire.KEY_LEN)
    port = free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "engine.stage_worker", "--port", str(port),
         "--layers", "12:24", "--model", MODEL, "--key", key.hex(), "--device", "cpu"],
        cwd=REPO)
    print("scheduler over co-located 0:12 + remote 12:24")

    coord = None
    try:
        wait_listen(port)
        coord = Coordinator(MODEL, [("127.0.0.1", port)], key, local_layers=(0, 12), device="cpu")

        # references first (scheduler idle, no connection contention)
        refs = [coord.generate(p, N_NEW)[1] for p in PROMPTS]

        sched = BatchScheduler(coord, max_batch=4, max_wait=0.1)
        t0 = time.time()
        out_qs = [sched.submit(p, N_NEW) for p in PROMPTS]   # submitted together -> one batch
        results = [drain(q) for q in out_qs]
        dt = time.time() - t0

        allok = True
        for b, p in enumerate(PROMPTS):
            ok = results[b] == refs[b]
            allok = allok and ok
            print(f"  [{'OK ' if ok else 'BAD'}] req {b} ({len(results[b])} tok): {coord.tok.decode(results[b])!r}")
            if not ok:
                print(f"        ref: {coord.tok.decode(refs[b])!r}")
        assert allok, "a scheduled request's stream diverged from its per-seq reference"
        print(f"B3 SCHEDULER PASSED — concurrent requests batched + streamed, each "
              f"token-identical to per-seq generate ({dt:.1f}s)")
        sched.stop()
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
