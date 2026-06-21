"""
test_failover_exhausted.py — CHAOS: failover with NOWHERE to fail over to.

R=2 on a single slot {b0,b1}. We stream, then KILL BOTH holders mid-stream. Failover
has no healthy replica left, so the route snapshot hits a coverage gap. The contract
under test: the request fails CLEANLY — it raises promptly (no hang), and every token
emitted before the failure was correct (no garbage from a half-dead pipeline). A
flaky mesh must degrade to a clean error, never to corruption.

Run on CPU:  python3 -m tests.test_failover_exhausted
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
from engine.topology import Topology, Node, READY  # noqa: E402
from engine.registry import Registry  # noqa: E402

MODEL = os.environ.get("CIRCUIT_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
PROMPT = "The capital of France is"
N_NEW = 20
KILL_AFTER = 5


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def wait_listen(port: int, timeout: float = 90.0):
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
    keyhex = key.hex()
    # one slot covering the whole model, two replicas — kill both and there is no
    # healthy holder left for that slot.
    nodes = {"b0": free_port(), "b1": free_port()}
    procs = {nid: subprocess.Popen(
        [sys.executable, "-m", "engine.stage_worker", "--port", str(port),
         "--layers", "0:24", "--model", MODEL, "--key", keyhex, "--device", "cpu"],
        cwd=REPO) for nid, port in nodes.items()}
    print("launched 2 replicas of slot 0:24 (R=2)")

    coord = None
    try:
        topo = Topology(num_layers=24, coordinator_end=0, num_stages=1,
                        model_fp="m", replication=2, dead_after_s=1e18)
        reg = Registry(topo=topo, master_secret=b"x", coordinator_endpoint=("c", 0))
        for nid, port in nodes.items():
            node = Node(node_id=nid, endpoint=("127.0.0.1", port), capacity_layers=24,
                        model_fp="m", state=READY)
            node.slot = 0
            topo.nodes[nid] = node
            topo.slots[0].holders.append(nid)
        for nid, hb in [("b1", 1.0), ("b0", 2.0)]:
            topo.heartbeat(nid, hb)
        for port in nodes.values():
            wait_listen(port)

        coord = Coordinator(MODEL, [], key, registry=reg, device="cpu")

        # clean reference (both alive)
        ref = ""
        for delta in coord.generate_stream(PROMPT, N_NEW, stop_on_eos=False):
            ref += delta
        print(f"  clean reference: {ref!r}")

        # killed run: kill BOTH holders mid-stream; expect a clean raise
        partial = ""
        raised = None
        t0 = time.time()
        try:
            for i, delta in enumerate(coord.generate_stream(PROMPT, N_NEW, stop_on_eos=False)):
                partial += delta
                if i == KILL_AFTER:
                    for nid in ("b0", "b1"):
                        procs[nid].kill()
                    for nid in ("b0", "b1"):
                        procs[nid].wait(timeout=5)
                    print("  >>> KILLED both holders mid-stream — no replica remains")
        except Exception as e:   # noqa: BLE001 — the point is that it surfaces, not hangs
            raised = e
        dt = time.time() - t0

        assert raised is not None, "generation must FAIL when a slot loses all holders, not return"
        assert dt < 60, f"failed but took too long ({dt:.1f}s) — looks like a hang, not a clean error"
        assert ref.startswith(partial), (
            f"tokens before the failure must be correct (no garbage): {partial!r} not a prefix of {ref!r}")
        assert all(p.poll() is not None for p in procs.values()), "both holders dead"
        print(f"  clean failure: {type(raised).__name__}: {str(raised)[:80]!r}  "
              f"(after {len(partial)} good chars, {dt:.1f}s)")
        print("EXHAUSTED-FAILOVER PASSED — total slot loss raises cleanly, no hang, "
              "no garbage tokens before the failure")
    finally:
        if coord is not None:
            coord.close()
        for p in procs.values():
            p.terminate()
        for p in procs.values():
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()


if __name__ == "__main__":
    main()
