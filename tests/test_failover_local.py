"""
test_failover_local.py — CHAOS failover with a CO-LOCATED local stage.

This matches the LIVE shape that test_failover skipped: the coordinator runs the
early layers (0:12) IN-PROCESS (its own KV) and the mesh serves the tail (12:24)
with replication R=2 {b0,b1}. We stream, KILL the pinned mesh holder b0 mid-stream,
and require the session to survive — which forces _reprefill to rebuild BOTH the
replica's KV AND the coordinator's own local-stage KV, then continue. Output must be
identical to a clean run through the same mesh.

Why it matters: the production coordinator co-locates a stage, so this is the path a
real node death would actually hit. The logic is meant to handle it (reset-all +
replay covers the local cache too); this proves it.

Run on CPU:  python3 -m tests.test_failover_local
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
from engine.topology import Topology, Node, READY, SUSPECT  # noqa: E402
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


def _stream(coord, kill=None):
    """Consume a full streamed greedy generation, optionally invoking kill() after
    KILL_AFTER tokens. Returns the accumulated text."""
    out = ""
    for i, delta in enumerate(coord.generate_stream(PROMPT, N_NEW, stop_on_eos=False)):
        out += delta
        if kill is not None and i == KILL_AFTER:
            kill()
            kill = None
    return out


def main():
    key = os.urandom(wire.KEY_LEN)
    keyhex = key.hex()
    # two replicas of the mesh tail slot (12:24); the coordinator co-locates 0:12
    nodes = {"b0": free_port(), "b1": free_port()}
    procs = {nid: subprocess.Popen(
        [sys.executable, "-m", "engine.stage_worker", "--port", str(port),
         "--layers", "12:24", "--model", MODEL, "--key", keyhex, "--device", "cpu"],
        cwd=REPO) for nid, port in nodes.items()}
    print("launched 2 mesh holders for 12:24 (R=2); coordinator will co-locate 0:12")

    coord = None
    try:
        topo = Topology(num_layers=24, coordinator_end=12, num_stages=1,
                        model_fp="m", replication=2, dead_after_s=1e18)
        reg = Registry(topo=topo, master_secret=b"x", coordinator_endpoint=("c", 0))
        for nid, port in nodes.items():
            node = Node(node_id=nid, endpoint=("127.0.0.1", port), capacity_layers=12,
                        model_fp="m", state=READY)
            node.slot = 0
            topo.nodes[nid] = node
            topo.slots[0].holders.append(nid)
        for nid, hb in [("b1", 1.0), ("b0", 2.0)]:     # b0 = primary (freshest)
            topo.heartbeat(nid, hb)
        assert [n.node_id for n in reg.route_snapshot()] == ["b0"], "b0 should be primary"

        for port in nodes.values():
            wait_listen(port)

        coord = Coordinator(MODEL, [], key, local_layers=(0, 12), registry=reg, device="cpu")
        assert coord.local_stage is not None, "coordinator must have a co-located stage"

        ref = _stream(coord)                            # clean run through b0
        print(f"  clean reference: {ref!r}")

        killed = {"done": False}

        def kill_b0():
            procs["b0"].kill()
            procs["b0"].wait(timeout=5)
            killed["done"] = True
            print(f"  >>> KILLED mesh holder b0 mid-stream — coordinator (local 0:12) "
                  f"must fail over to b1 and rebuild local + remote KV")

        out = _stream(coord, kill=kill_b0)

        assert killed["done"] and procs["b0"].poll() is not None, "b0 should be dead"
        assert topo.nodes["b0"].state == SUSPECT, (
            f"dead holder should be SUSPECT — state={topo.nodes['b0'].state}")
        match = out == ref
        print(f"  survived-failover decode: {match}")
        print(f"  output: {out!r}")
        assert match, "output after failover diverged from the clean run"
        print("LOCAL-STAGE FAILOVER PASSED — co-located 0:12 + mesh 12:24, killed the "
              "holder mid-stream, _reprefill rebuilt local + remote KV, output matches")
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
