"""
test_failover_hung.py — CHAOS: a node that HANGS (alive, accepts, never replies).

Death is handled (RST -> fast-fail -> failover). A *hung* node is the nastier case: the
socket stays open, the coordinator's read blocks. Without a read timeout it blocks
FOREVER and the request never completes. This test puts a hung fake-node as the pinned
primary of a slot (R=2) with a real worker as the replica, and requires the session to
TIME OUT the hung hop and fail over — finishing correctly, not hanging.

Run on CPU:  python3 -m tests.test_failover_hung
  (CIRCUIT_STAGE_READ_TIMEOUT is set low here so the test doesn't wait the default.)
"""

import os
import socket
import subprocess
import sys
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

os.environ.setdefault("CIRCUIT_STAGE_READ_TIMEOUT", "4")   # fail a hung hop fast in the test

from engine import wire  # noqa: E402
from engine.coordinator import Coordinator  # noqa: E402
from engine.topology import Topology, Node, READY, SUSPECT  # noqa: E402
from engine.registry import Registry  # noqa: E402

MODEL = os.environ.get("CIRCUIT_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
PROMPT = "The capital of France is"
N_NEW = 16


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


def hung_server(port, stop):
    """Accept connections and read forever, NEVER replying — a stalled node."""
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(8)
    srv.settimeout(0.5)
    conns = []
    while not stop.is_set():
        try:
            c, _ = srv.accept()
            conns.append(c)
        except socket.timeout:
            continue
        except OSError:
            break
    for c in conns:
        try:
            c.close()
        except OSError:
            pass
    srv.close()


def main():
    key = os.urandom(wire.KEY_LEN)
    hung_port = free_port()
    good_port = free_port()
    stop = threading.Event()
    threading.Thread(target=hung_server, args=(hung_port, stop), daemon=True).start()
    good = subprocess.Popen(
        [sys.executable, "-m", "engine.stage_worker", "--port", str(good_port),
         "--layers", "0:24", "--model", MODEL, "--key", key.hex(), "--device", "cpu"],
        cwd=REPO)
    print(f"hung fake-node :{hung_port} (primary)  +  real replica :{good_port}")

    coord = None
    try:
        topo = Topology(num_layers=24, coordinator_end=0, num_stages=1,
                        model_fp="m", replication=2, dead_after_s=1e18)
        reg = Registry(topo=topo, master_secret=b"x", coordinator_endpoint=("c", 0))
        for nid, port in [("hung", hung_port), ("good", good_port)]:
            node = Node(node_id=nid, endpoint=("127.0.0.1", port), capacity_layers=24,
                        model_fp="m", state=READY)
            node.slot = 0
            topo.nodes[nid] = node
            topo.slots[0].holders.append(nid)
        for nid, hb in [("good", 1.0), ("hung", 2.0)]:   # hung = primary (freshest)
            topo.heartbeat(nid, hb)
        assert [n.node_id for n in reg.route_snapshot()] == ["hung"], "hung is primary"

        wait_listen(hung_port)
        wait_listen(good_port)
        coord = Coordinator(MODEL, [], key, registry=reg, device="cpu")

        t0 = time.time()
        text, toks = coord.generate(PROMPT, N_NEW)
        dt = time.time() - t0

        assert dt < 60, f"took {dt:.1f}s — the hung hop was NOT timed out (it blocked)"
        assert topo.nodes["hung"].state == SUSPECT, \
            f"hung node should be SUSPECT after the timeout — got {topo.nodes['hung'].state}"
        assert len(toks) == N_NEW and text.strip(), "failover should have produced the full output"
        print(f"  output: {text!r}")
        print(f"HUNG-NODE FAILOVER PASSED — timed out the stalled primary in ~{dt:.1f}s, "
              f"failed over to the live replica, produced correct output")
    finally:
        stop.set()
        if coord is not None:
            coord.close()
        good.terminate()
        try:
            good.wait(timeout=5)
        except subprocess.TimeoutExpired:
            good.kill()


if __name__ == "__main__":
    main()
