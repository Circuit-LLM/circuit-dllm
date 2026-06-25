"""
test_rehome.py — an orchestrator dies mid-session; another RE-HOMES it with NO re-prefill.

This is the elegant half of the floating coordinator (docs/FLOATING_COORDINATOR.md §6): layer KV
lives on the slice-holders, keyed by session-id, in a WORKER-GLOBAL store — so it outlives the
orchestrator that built it. A second orchestrator picks the session up, re-acquires the same route,
and continues from the carried head-side state WITHOUT re-running the prompt.

Setup: 3 real stage workers (head/middle/tail over 0.5B, CPU), a shared registry, two independent
head-only orchestrators A and B (no co-located slice, LocalRouteProvider over the shared registry).

  • ground truth   = a single-model greedy reference of N = K + M tokens
  • orchestrator A drives session S for the first K tokens, then DROPS its connections (crash) —
    the workers keep S's KV in the global store
  • orchestrator B re-homes S and generates the next M tokens via generate_resume (no pos==0)
  • assert  A's K  ++  B's M  ==  reference     → B reused A's KV; if it had re-prefilled (or the KV
    were gone) the continuation would diverge.

Negative control: re-home a session-id the workers never saw → empty KV → the continuation MUST
diverge. That proves the positive result comes from genuine KV reuse, not coincidence.

  python3 -m tests.test_rehome          # run on a pod (needs torch); CPU is fine
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
K = 8        # tokens orchestrator A produces before it "dies"
M = 12       # tokens orchestrator B produces after re-homing
N = K + M


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


def _register_holders(reg, topo, ports, keys):
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


def main():
    keys = [os.urandom(wire.KEY_LEN) for _ in range(3)]
    coord_key = os.urandom(wire.KEY_LEN)
    ports = [free_port(), free_port(), free_port()]
    layer_ranges = ["0:8", "8:16", "16:24"]

    procs = []
    for port, layers, k in zip(ports, layer_ranges, keys):
        procs.append(subprocess.Popen(
            [sys.executable, "-m", "engine.stage_worker", "--port", str(port),
             "--layers", layers, "--model", MODEL, "--key", k.hex(), "--device", "cpu"],
            cwd=REPO))
    print(f"launched 3 stage workers on {ports}")

    A = B = None
    try:
        topo = Topology(num_layers=24, coordinator_end=0, num_stages=3,
                        model_fp="m", replication=1, dead_after_s=1e18)
        reg = Registry(topo=topo, master_secret=b"x", coordinator_endpoint=("c", 0))
        _register_holders(reg, topo, ports, keys)

        S = 7777                                       # the (globally-unique) session id

        # ── orchestrator A: head-only, drives session S for K tokens, then crashes ──
        A = Coordinator(MODEL, [], coord_key, local_layers=None, registry=reg)
        ref = _reference(A._model, A.tok, N)
        ids = A.tok(PROMPT, return_tensors="pt").input_ids.to(A.device)
        seq = ids.shape[1]
        hidden = A._relay_with_failover(S, 0, A.embed(ids), ids[:, :0])     # prefill the prompt
        nxt = A.lm_head(A.norm(hidden))[:, -1].argmax(-1, keepdim=True)
        out_A, seq_ids, nxt, cur = A._greedy_steps(S, ids, nxt, seq, K)     # K tokens; capture state
        assert cur == seq + K
        A.close()                                       # CRASH: drop A's sockets (no KV reset/free)
        print(f"  A produced {len(out_A)} tokens, then dropped its connections")

        # ── orchestrator B: head-only, re-homes session S and continues for M tokens ──
        B = Coordinator(MODEL, [], coord_key, local_layers=None, registry=reg)
        _, out_B = B.generate_resume(S, seq_ids, nxt, cur, M)
        print(f"  B re-homed session {S} and produced {len(out_B)} more tokens")

        ok_A = out_A == ref[:K]
        ok_B = out_B == ref[K:N]
        print(f"  A == reference[:K] : {ok_A}")
        print(f"  B == reference[K:N]: {ok_B}   (continuation reused the holders' KV — no re-prefill)")
        if not (ok_A and ok_B):
            print("  ref:", ref)
            print("  A+B:", out_A + out_B)
        assert ok_A, "orchestrator A diverged from the reference (harness problem)"
        assert ok_B, "re-home DIVERGED — B did not reuse the session KV (re-prefill or KV lost)"

        # ── negative control: re-home a session the workers never built → empty KV → must diverge ──
        _, out_neg = B.generate_resume(999999, seq_ids, nxt, cur, M)
        assert out_neg != ref[K:N], ("negative control matched the reference — the positive result "
                                     "did NOT depend on persisted KV, so it proves nothing")
        print(f"  negative control (fresh session, empty KV) diverged as expected: "
              f"{out_neg[:3]}... != {ref[K:K+3]}...")

        print("RE-HOME TEST PASSED — a second orchestrator continued a dead orchestrator's session "
              "byte-identically with NO re-prefill (worker-global KV); empty-KV control diverged")
    finally:
        for c in (A, B):
            if c is not None:
                try:
                    c.close()
                except Exception:
                    pass
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()


if __name__ == "__main__":
    main()
