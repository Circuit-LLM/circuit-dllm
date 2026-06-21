"""
test_failover.py — CHAOS: a holder dies mid-generation, the session survives.

A 24-layer model is served by a mesh with replication R=2: slot0 (layers 0:12) has
two holders {a0,a1}, slot1 (layers 12:24) has {b0,b1}. The coordinator pins a0,b0
and starts streaming. Mid-stream we KILL a0 (the pinned primary). The next hop
fails; the coordinator marks a0 SUSPECT, re-prefills the session's KV on the
replica a1, and finishes — producing output token-for-token identical to a clean
in-process run.

This is the reliability gate: it proves a flaky stranger node cannot corrupt or
abort an in-flight request once R>=2.

Run on CPU:  python3 -m tests.test_failover
"""

import os
import socket
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from transformers.cache_utils import DynamicCache  # noqa: E402

from engine import wire  # noqa: E402
from engine.coordinator import Coordinator  # noqa: E402
from engine.topology import Topology, Node, READY, SUSPECT  # noqa: E402
from engine.registry import Registry  # noqa: E402

MODEL = os.environ.get("CIRCUIT_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
PROMPT = "The capital of France is"
N_NEW = 20
KILL_AFTER = 5   # tokens streamed before we kill the primary


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
    # 4 workers: two replicas per slot (cluster key — registration-issued keys are
    # exercised by test_node_join; here we focus purely on the failover mechanism).
    nodes = {"a0": ("0:12", free_port()), "a1": ("0:12", free_port()),
             "b0": ("12:24", free_port()), "b1": ("12:24", free_port())}
    procs = {}
    for nid, (layers, port) in nodes.items():
        procs[nid] = subprocess.Popen(
            [sys.executable, "-m", "engine.stage_worker", "--port", str(port),
             "--layers", layers, "--model", MODEL, "--key", keyhex, "--device", "cpu"],
            cwd=REPO)
    print("launched 4 stage workers (R=2): slot0={a0,a1} slot1={b0,b1}")

    coord = None
    try:
        topo = Topology(num_layers=24, coordinator_end=0, num_stages=2,
                        model_fp="m", replication=2, dead_after_s=1e18)
        reg = Registry(topo=topo, master_secret=b"x", coordinator_endpoint=("c", 0))
        for nid, (_layers, port) in nodes.items():
            slot_idx = 0 if nid.startswith("a") else 1
            node = Node(node_id=nid, endpoint=("127.0.0.1", port), capacity_layers=12,
                        model_fp="m", state=READY)
            node.slot = slot_idx
            topo.nodes[nid] = node
            topo.slots[slot_idx].holders.append(nid)
        # a0,b0 are PRIMARY (freshest heartbeat); a1,b1 are the failover replicas.
        for nid, hb in [("a1", 1.0), ("b1", 1.0), ("a0", 2.0), ("b0", 2.0)]:
            topo.heartbeat(nid, hb)
        prim = [n.node_id for n in reg.route_snapshot()]
        assert prim == ["a0", "b0"], f"primaries should be a0,b0 — got {prim}"

        for _nid, (_layers, port) in nodes.items():
            wait_listen(port)

        coord = Coordinator(MODEL, [], key, registry=reg)
        ref = _reference(coord._model, coord.tok, N_NEW)
        ref_text = coord.tok.decode(ref)

        out = ""
        killed = False
        for i, delta in enumerate(coord.generate_stream(PROMPT, N_NEW, stop_on_eos=False)):
            out += delta
            if i == KILL_AFTER and not killed:
                procs["a0"].kill()
                procs["a0"].wait(timeout=5)
                killed = True
                print(f"  >>> KILLED primary a0 after {i + 1} tokens — "
                      f"coordinator must fail over to replica a1")

        assert killed, "never reached the kill point"
        assert procs["a0"].poll() is not None, "a0 should be dead"
        assert topo.nodes["a0"].state == SUSPECT, (
            "coordinator should have marked the dead primary SUSPECT "
            f"(failover path) — state={topo.nodes['a0'].state}")
        match = out == ref_text
        print(f"  survived-failover decode: {match}  ({N_NEW} tokens)")
        print(f"  output: {out!r}")
        if not match:
            print("  ref:", repr(ref_text))
        assert match, "output after failover diverged from the clean reference"
        print("FAILOVER CHAOS TEST PASSED — killed the pinned primary mid-stream, "
              "session re-prefilled on the replica and finished correctly")
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
