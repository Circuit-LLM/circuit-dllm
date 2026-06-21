"""
test_node_join.py — END TO END node join.

Two stage-worker processes JOIN a coordinator's mesh over the HTTP control channel:
each registers (gets a per-node key + a layer assignment), loads its layers, serves,
and heartbeats. Then the coordinator generates by routing through them — using the
SAME per-node keys the registry issued. Proves self-service join + per-node keys +
dynamic routing working together over real sockets.

Run on CPU:  python3 -m tests.test_node_join
"""

import os
import socket
import subprocess
import sys
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch  # noqa: E402
from transformers.cache_utils import DynamicCache  # noqa: E402

from engine import wire  # noqa: E402
from engine.coordinator import Coordinator  # noqa: E402
from engine.topology import Topology  # noqa: E402
from engine.registry import Registry  # noqa: E402
from engine.control_server import make_server  # noqa: E402

MODEL = os.environ.get("CIRCUIT_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
PROMPT = "The capital of France is"
N_NEW = 20
FP = "test-fp"


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
    # control server over a 24-layer model: coordinator holds none, two stages.
    topo = Topology(num_layers=24, coordinator_end=0, num_stages=2,
                    model_fp=FP, replication=1, dead_after_s=1e18)
    reg = Registry(topo=topo, master_secret=os.urandom(16),
                   coordinator_endpoint=["127.0.0.1", 0], allowlist={"node-a", "node-b"})
    cport = free_port()
    srv = make_server(reg, host="127.0.0.1", port=cport, reap_interval=1e9)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    control_url = f"http://127.0.0.1:{cport}"

    p_a, p_b = free_port(), free_port()
    procs = []
    for nid, port in [("node-a", p_a), ("node-b", p_b)]:
        procs.append(subprocess.Popen(
            [sys.executable, "-m", "engine.stage_worker", "--port", str(port),
             "--model", MODEL, "--device", "cpu", "--host", "127.0.0.1",
             "--control-url", control_url, "--node-id", nid,
             "--capacity-layers", "12", "--model-fp", FP,
             "--advertise-host", "127.0.0.1", "--hb-interval", "2"],
            cwd=REPO))
    print(f"launched 2 nodes joining via {control_url}")

    coord = None
    try:
        # wait for both to register, load, and report READY (full coverage)
        ready = False
        for _ in range(180):
            if topo.coverage_ok():
                ready = True
                break
            time.sleep(1)
        assert ready, "nodes did not join + become READY in time"
        # they joined with DERIVED per-node keys (issued by registry.register)
        assert all(n.wire_key for n in topo.nodes.values()), "nodes hold per-node keys"
        assert {n.node_id: (n.slot) for n in topo.nodes.values()} == {"node-a": 0, "node-b": 1} \
            or {n.node_id: (n.slot) for n in topo.nodes.values()} == {"node-a": 1, "node-b": 0}, "one node per slot"

        # the coordinator's cluster key is irrelevant here — it talks to each node
        # with that node's issued key (via node.wire_key).
        coord = Coordinator(MODEL, [], wire.normalize_key(os.urandom(32)), registry=reg)
        t0 = time.time()
        text, got = coord.generate(PROMPT, N_NEW)
        dt = time.time() - t0
        ref = _reference(coord._model, coord.tok, N_NEW)

        match = got == ref
        print(f"  joined-mesh decode: {match}  ({N_NEW} tokens, {dt:.1f}s)")
        print(f"  output: {text!r}")
        assert match, "joined-mesh decode diverged from the in-process reference"
        print("NODE-JOIN E2E PASSED — nodes joined over HTTP, coordinator routed via per-node keys")
    finally:
        if coord is not None:
            coord.close()
        srv.shutdown()
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()


if __name__ == "__main__":
    main()
