"""
test_dynamic_relay.py — the coordinator routing through the dynamic Topology/Registry
produces output identical to an in-process reference, proving the mesh control plane
drives real generation on the encrypted socket path (not just the static stage list).

Mirrors test_twoproc, but the coordinator is in DYNAMIC mode: two real stage workers
are registered as the mesh's holders and routing goes through Topology.route().

Run on CPU:  python3 -m tests.test_dynamic_relay
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
N_NEW = 20


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
    key = os.urandom(wire.KEY_LEN)
    keyhex = key.hex()
    p0, p1 = free_port(), free_port()

    procs = []
    for port, layers in [(p0, "0:12"), (p1, "12:24")]:
        procs.append(subprocess.Popen(
            [sys.executable, "-m", "engine.stage_worker", "--port", str(port),
             "--layers", layers, "--model", MODEL, "--key", keyhex, "--device", "cpu"],
            cwd=REPO))
    print(f"launched 2 stage workers on {p0}, {p1}")

    coord = None
    try:
        # Build the live mesh: 24-layer model, coordinator holds none, two slots.
        # Register the two workers as READY holders. wire_key=None → the coordinator
        # talks to them with the cluster key (matching their --key); in production
        # registration issues each a derived key instead.
        topo = Topology(num_layers=24, coordinator_end=0, num_stages=2,
                        model_fp="m", replication=1, dead_after_s=1e18)
        reg = Registry(topo=topo, master_secret=b"x", coordinator_endpoint=("c", 0))
        for nid, port, slot_idx in [("s0", p0, 0), ("s1", p1, 1)]:
            node = Node(node_id=nid, endpoint=("127.0.0.1", port), capacity_layers=12,
                        model_fp="m", state=READY)
            node.slot = slot_idx
            topo.nodes[nid] = node
            topo.slots[slot_idx].holders.append(nid)
            topo.heartbeat(nid, 1.0)
        assert topo.coverage_ok(), "mesh not fully covered"
        # routing plan must be the two slots in order
        plan = topo.route()
        assert [(s.start, s.end) for s, _ in plan] == [(0, 12), (12, 24)]

        coord = Coordinator(MODEL, [], key, registry=reg)   # DYNAMIC mode
        t0 = time.time()
        text, got = coord.generate(PROMPT, N_NEW)
        dt = time.time() - t0
        ref = _reference(coord._model, coord.tok, N_NEW)

        match = got == ref
        print(f"  dynamic-mesh decode: {match}  ({N_NEW} tokens, {dt:.1f}s)")
        print(f"  output: {text!r}")
        if not match:
            print("  ref:", ref)
            print("  got:", got)
        assert match, "dynamic-mesh decode diverged from the in-process reference"
        print("DYNAMIC-RELAY TEST PASSED — Topology-routed generation matches reference")
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
