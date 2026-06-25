"""
test_remote_route.py — a HEAD-ONLY orchestrator over a REMOTE control plane is byte-identical.

This is the floating-coordinator data path (docs/FLOATING_COORDINATOR.md §4b, §5). The orchestrator:
  • holds ONLY the head bundle (embed / norm / lm_head) — no co-located layer slice,
  • owns NO local registry — it gets its route over the authenticated control channel
    (RemoteRouteProvider → control_server → registry.acquire_route), exactly as a real head-only
    node on a scattered mesh would.

It drives THREE real stage workers (head/middle/tail over 0.5B, CPU) and must produce output
BYTE-IDENTICAL to the in-process reference. That proves the registry-optional relay path
(provider-driven _dynamic, RouteHop as a drop-in for Node, route-key cache, provider.mark_suspect,
tree-via-provider) is correct against REAL workers, not mocks. The in-process Coordinator path is
covered separately by test_chain_relay (still byte-identical — the local path is unchanged).

  python3 -m tests.test_remote_route          # run on a pod (needs torch); CPU is fine
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
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # noqa: E402
from cryptography.hazmat.primitives.serialization import (  # noqa: E402
    Encoding, PrivateFormat, PublicFormat, NoEncryption)

from engine import wire  # noqa: E402
from engine.coordinator import Coordinator  # noqa: E402
from engine.model import load_model  # noqa: E402
from engine.topology import Topology, Node, READY  # noqa: E402
from engine.registry import Registry  # noqa: E402
from engine.control_server import make_server, make_ed25519_verifier, make_ed25519_signer  # noqa: E402
from engine.route_provider import RemoteRouteProvider  # noqa: E402

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


def _ed25519_pair():
    p = Ed25519PrivateKey.generate()
    priv = p.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption()).hex()
    pub = p.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
    return priv, pub


def main():
    keys = [os.urandom(wire.KEY_LEN) for _ in range(3)]   # distinct per-node data-wire keys
    coord_key = os.urandom(wire.KEY_LEN)                   # orchestrator's cluster key (fallback only)
    ports = [free_port(), free_port(), free_port()]
    layer_ranges = ["0:8", "8:16", "16:24"]               # 24-layer 0.5B → head / middle / tail

    procs = []
    for port, layers, k in zip(ports, layer_ranges, keys):
        procs.append(subprocess.Popen(
            [sys.executable, "-m", "engine.stage_worker", "--port", str(port),
             "--layers", layers, "--model", MODEL, "--key", k.hex(), "--device", "cpu"],
            cwd=REPO))
    print(f"launched 3 stage workers on {ports}")

    srv = coord = None
    try:
        # ── the control plane: a real registry + control_server, ed25519-authed, in a thread ──
        topo = Topology(num_layers=24, coordinator_end=0, num_stages=3,
                        model_fp="m", replication=1, dead_after_s=1e18)
        reg = Registry(topo=topo, master_secret=b"x", coordinator_endpoint=("c", 0))
        for (nid, port, slot_idx), k in zip(
                [("s0", ports[0], 0), ("s1", ports[1], 1), ("s2", ports[2], 2)], keys):
            node = Node(node_id=nid, endpoint=("127.0.0.1", port), capacity_layers=8,
                        model_fp="m", state=READY)
            node.slot = slot_idx
            node.wire_key = k
            topo.nodes[nid] = node
            topo.slots[slot_idx].holders.append(nid)
            topo.heartbeat(nid, 1.0)
        assert topo.coverage_ok(), "mesh not fully covered"

        srv = make_server(reg, host="127.0.0.1", port=0, reap_interval=3600,
                          verify_sig=make_ed25519_verifier())
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{srv.server_address[1]}"

        # ── the head-only orchestrator: no local slice, no local registry, route via the control plane ──
        priv, pub = _ed25519_pair()
        rp = RemoteRouteProvider(url, make_ed25519_signer(priv, pub))
        # CIRCUIT_TEST_SHARD=1 exercises the HEAD-ONLY SHARD load path (the 72B orchestrator path:
        # embed/norm/lm_head only, layers on meta) instead of a whole fp16 load — must be byte-identical.
        shard = os.environ.get("CIRCUIT_TEST_SHARD") == "1"
        coord = Coordinator(MODEL, [], coord_key, local_layers=None, shard=shard, route_provider=rp)
        print(f"  (head-only load: {'SHARD' if shard else 'whole-fp16'})")
        assert coord.registry is None, "orchestrator must NOT hold a local registry"
        assert coord._dynamic, "orchestrator must route dynamically (via the provider)"
        assert coord.local_stage is None, "head-only: no co-located layer slice"

        # the reference is a FULL-model forward; coord._model is head-only when sharded (layers
        # dropped to meta/Identity), so load a separate whole model for the ground truth.
        ref_model = load_model(MODEL) if shard else coord._model
        ref = _reference(ref_model, coord.tok, N_NEW)
        t0 = time.time(); text, got = coord.generate(PROMPT, N_NEW); dt = time.time() - t0

        print(f"  head-only remote-route: {got == ref}  ({dt:.1f}s)")
        print(f"  output: {text!r}")
        if got != ref:
            print("  ref :", ref)
            print("  got :", got)
        assert got == ref, "head-only remote-route diverged from the in-process reference"
        # the route-key cache learned every holder over the authenticated channel (no local registry)
        assert set(coord._route_keys) == {"s0", "s1", "s2"}, "route-key cache did not learn the holders"
        print("REMOTE-ROUTE TEST PASSED — head-only orchestrator over a remote control plane is "
              "byte-identical to the reference (registry-optional relay correct)")
    finally:
        if coord is not None:
            coord.close()
        if srv is not None:
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
