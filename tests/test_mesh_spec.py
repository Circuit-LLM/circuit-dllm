"""
test_mesh_spec.py — predictive drafting (speculative decode) over the DYNAMIC mesh path.

The live engine uses speculative decode; the mesh/dynamic route had only been tested
with greedy. This closes that gap before converting the live 2 L4s to the mesh: a
dynamic-mode coordinator (registry-routed, per the node-join control plane) runs
generate_speculative through two registered stage workers — including the KV rollback
(KV_CTRL truncate) over the dynamic connections — and the output must equal plain
greedy. If this passes, the live mesh can keep drafting on.

Run on CPU:  python3 -m tests.test_mesh_spec
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
from engine.topology import Topology, Node, READY  # noqa: E402
from engine.registry import Registry  # noqa: E402

MODEL = os.environ.get("CIRCUIT_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
PROMPT = "The capital of France is"
N_NEW = 24
K = 4


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


def _greedy_ref(model, tok, n_new):
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
    p0, p1 = free_port(), free_port()
    procs = [subprocess.Popen(
        [sys.executable, "-m", "engine.stage_worker", "--port", str(port),
         "--layers", layers, "--model", MODEL, "--key", key.hex(), "--device", "cpu"],
        cwd=REPO) for port, layers in [(p0, "0:12"), (p1, "12:24")]]
    print("speculative decode over the DYNAMIC mesh route (2 registered stage workers)")

    coord = None
    try:
        topo = Topology(num_layers=24, coordinator_end=0, num_stages=2,
                        model_fp="m", replication=1, dead_after_s=1e18)
        reg = Registry(topo=topo, master_secret=b"x", coordinator_endpoint=("c", 0))
        for nid, port, slot in [("s0", p0, 0), ("s1", p1, 1)]:
            node = Node(node_id=nid, endpoint=("127.0.0.1", port), capacity_layers=12,
                        model_fp="m", state=READY)
            node.slot = slot
            topo.nodes[nid] = node
            topo.slots[slot].holders.append(nid)
            topo.heartbeat(nid, 1.0)
        assert topo.coverage_ok()

        wait_listen(p0)
        wait_listen(p1)
        coord = Coordinator(MODEL, [], key, registry=reg, device="cpu")   # DYNAMIC mode

        t0 = time.time()
        text, got = coord.generate_speculative(PROMPT, N_NEW, K=K)
        dt = time.time() - t0
        ref = _greedy_ref(coord._model, coord.tok, N_NEW)

        match = got == ref
        print(f"  mesh speculative decode: {match}  ({N_NEW} tokens, {dt:.1f}s)")
        print(f"  output: {text!r}")
        print(f"  spec_stats: {coord.spec_stats()}")
        if not match:
            print("  ref:", ref)
            print("  got:", got)
        assert match, "speculative-over-mesh diverged from greedy — dynamic rollback path is wrong"
        print("MESH-SPECULATIVE PASSED — predictive drafting over the dynamic route is "
              "token-identical to greedy (rollback over dynamic conns works)")
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
