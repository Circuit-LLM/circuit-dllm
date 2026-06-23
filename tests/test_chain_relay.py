"""
test_chain_relay.py — chain relay produces output BYTE-IDENTICAL to the star.

Spins up THREE real stage workers (so the chain has a head, a middle, and a tail — the
middle exercises the forward-and-bubble path), registers them as the mesh holders, then
runs the SAME greedy generation two ways through one coordinator:

  • star  (CIRCUIT_CHAIN off) — coordinator round-trips to every stage
  • chain (CIRCUIT_CHAIN on)  — coordinator → head → middle → tail, result bubbles back

and asserts  star == in-process reference == chain.  Same nodes, same pinned order, same
KV — only the data path differs, so the tokens must match exactly. See docs/CHAIN_RELAY.md.

Needs torch + a small model → run on a pod (CPU is fine), NOT the VPS:
  python3 -m tests.test_chain_relay
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
    ports = [free_port(), free_port(), free_port()]
    layer_ranges = ["0:8", "8:16", "16:24"]      # 24-layer 0.5B → head / middle / tail

    procs = []
    for port, layers in zip(ports, layer_ranges):
        procs.append(subprocess.Popen(
            [sys.executable, "-m", "engine.stage_worker", "--port", str(port),
             "--layers", layers, "--model", MODEL, "--key", keyhex, "--device", "cpu"],
            cwd=REPO))
    print(f"launched 3 stage workers on {ports}")

    coord = None
    try:
        topo = Topology(num_layers=24, coordinator_end=0, num_stages=3,
                        model_fp="m", replication=1, dead_after_s=1e18)
        reg = Registry(topo=topo, master_secret=b"x", coordinator_endpoint=("c", 0))
        # Register the 3 workers as READY holders, one per slot. CRUCIAL for the chain:
        # set each node's wire_key = the cluster key, since all workers framed with --key.
        # chain_head_and_route encodes each downstream node's key into the route so the
        # PREVIOUS node can encrypt the forward to it.
        for nid, port, slot_idx in [("s0", ports[0], 0), ("s1", ports[1], 1), ("s2", ports[2], 2)]:
            node = Node(node_id=nid, endpoint=("127.0.0.1", port), capacity_layers=8,
                        model_fp="m", state=READY)
            node.slot = slot_idx
            node.wire_key = key
            topo.nodes[nid] = node
            topo.slots[slot_idx].holders.append(nid)
            topo.heartbeat(nid, 1.0)
        assert topo.coverage_ok(), "mesh not fully covered"
        assert [(s.start, s.end) for s, _ in topo.route()] == [(0, 8), (8, 16), (16, 24)]

        coord = Coordinator(MODEL, [], key, registry=reg)      # DYNAMIC mode, star by default
        ref = _reference(coord._model, coord.tok, N_NEW)

        coord._chain = False
        t0 = time.time(); _, got_star = coord.generate(PROMPT, N_NEW); dt_star = time.time() - t0

        coord._chain = True                                    # flip to chain relay
        t0 = time.time(); text, got_chain = coord.generate(PROMPT, N_NEW); dt_chain = time.time() - t0

        print(f"  star : {got_star == ref}  ({dt_star:.1f}s)")
        print(f"  chain: {got_chain == ref}  ({dt_chain:.1f}s)")
        print(f"  output: {text!r}")
        if got_star != ref or got_chain != ref:
            print("  ref  :", ref)
            print("  star :", got_star)
            print("  chain:", got_chain)
        assert got_star == ref, "star diverged from reference (harness problem)"
        assert got_chain == ref, "CHAIN diverged from reference"
        assert got_chain == got_star, "chain != star"
        print("CHAIN-RELAY TEST PASSED — chain output byte-identical to star + reference "
              "(head/middle/tail forwarding correct)")
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
