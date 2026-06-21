"""
test_stage_resilient.py — a stalled / half-open peer must NOT block a new peer.

Reproduces the wedge that took the stage worker down in production: the prior
single-connection accept loop blocked forever on a dead coordinator socket
(left half-open through the RunPod proxy when pod 1 restarted) and refused every
new connection. With one handler thread per peer, a new coordinator can always
connect and be served while the stale peer sits parked.

Run on CPU:  python3 -m tests.test_stage_resilient
"""

import os
import socket
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from engine import wire  # noqa: E402

MODEL = os.environ.get("CIRCUIT_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def main():
    key = wire.normalize_key(os.urandom(wire.KEY_LEN))
    port = free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "engine.stage_worker", "--port", str(port),
         "--layers", "0:12", "--model", MODEL, "--key", key.hex(), "--device", "cpu"],
        cwd=REPO)
    a = b = None
    try:
        # wait for the worker to bind
        ready = False
        for _ in range(180):
            try:
                socket.create_connection(("127.0.0.1", port), timeout=2).close()
                ready = True
                break
            except OSError:
                time.sleep(1)
        assert ready, "stage worker never came up"

        # peer A: connect and STALL — never send a full frame, so its handler
        # blocks reading. Under the old single-connection loop this wedges the
        # whole worker.
        a = socket.create_connection(("127.0.0.1", port), timeout=5)
        a.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        time.sleep(1.0)

        # peer B: must still be served promptly despite A stalling.
        b = socket.create_connection(("127.0.0.1", port), timeout=5)
        b.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        wire.write_frame(b, key, wire.PING, b"ping")
        b.settimeout(8)
        mt, _ = wire.read_frame_keyed(b, key)
        assert mt == wire.PONG, f"expected PONG, got {wire.msg_name(mt)} (worker is wedged)"
        print("STAGE RESILIENT TEST PASSED — new peer served while a prior peer stalled")
    finally:
        for s in (a, b):
            if s is not None:
                try:
                    s.close()
                except OSError:
                    pass
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
