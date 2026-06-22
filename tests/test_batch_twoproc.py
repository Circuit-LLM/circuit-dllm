"""
test_batch_twoproc.py — Win B / B2: a batched forward over the real socket.

Extends B1 (in-process) to the wire: a fixed left-padded batch of ragged-length
sequences is decoded through TWO stage workers via the new BATCH_ACTIVATION frame
(hidden [B,T,D] + per-row position_ids + 2D padding mask), and every row must match
its per-sequence greedy reference. Proves the batched wire + the worker's batched
forward path. Fixed batch (no dynamic admit/evict — that's B3).

Run on CPU:  python3 -m tests.test_batch_twoproc
"""

import os
import socket
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from transformers.cache_utils import DynamicCache  # noqa: E402

from engine import wire  # noqa: E402
from engine.tensors import pack_batch_activation, unpack_activation  # noqa: E402

MODEL = os.environ.get("CIRCUIT_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
N_NEW = 24
PROMPTS = [
    "The capital of France is",
    "Two plus two equals",
    "The largest planet in the solar system is the gas giant",
    "Water",
]


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


def seq_ref(model, ids, n_new):
    """Per-sequence greedy reference (unsplit full model)."""
    cache = DynamicCache(config=model.config)
    nxt = model(ids, past_key_values=cache, use_cache=True).logits[:, -1].argmax(-1, keepdim=True)
    out = [int(nxt)]
    for _ in range(n_new - 1):
        nxt = model(nxt, past_key_values=cache, use_cache=True).logits[:, -1].argmax(-1, keepdim=True)
        out.append(int(nxt))
    return out


def relay_batch(socks, key, batch_id, pos, hidden, position_ids, attn):
    """Send one batched activation through the stage chain, return the final hidden."""
    h = hidden
    for s in socks:
        wire.write_frame(s, key, wire.BATCH_ACTIVATION,
                         pack_batch_activation(batch_id, pos, h, position_ids, attn))
        mt, payload = wire.read_frame_keyed(s, key)
        assert mt == wire.RESULT, wire.msg_name(mt)
        _, _, _, h = unpack_activation(payload)
    return h


def batch_decode_socket(model, socks, key, ids, attn, n_new):
    B = ids.shape[0]
    pos = (attn.long().cumsum(-1) - 1).clamp(min=0)
    h = relay_batch(socks, key, 1, 0, model.model.embed_tokens(ids), pos, attn)
    nxt = model.lm_head(model.model.norm(h))[:, -1].argmax(-1, keepdim=True)
    lengths = attn.long().sum(-1)
    out = [[int(nxt[b])] for b in range(B)]
    cur_attn = attn
    for step in range(n_new - 1):
        cur_attn = torch.cat([cur_attn, torch.ones(B, 1, dtype=attn.dtype)], dim=1)
        h = relay_batch(socks, key, 1, step + 1,
                        model.model.embed_tokens(nxt), lengths.unsqueeze(1), cur_attn)
        nxt = model.lm_head(model.model.norm(h))[:, -1].argmax(-1, keepdim=True)
        lengths = lengths + 1
        for b in range(B):
            out[b].append(int(nxt[b]))
    return out


def main():
    key = os.urandom(wire.KEY_LEN)
    p0, p1 = free_port(), free_port()
    procs = [subprocess.Popen(
        [sys.executable, "-m", "engine.stage_worker", "--port", str(port),
         "--layers", layers, "--model", MODEL, "--key", key.hex(), "--device", "cpu"],
        cwd=REPO) for port, layers in [(p0, "0:12"), (p1, "12:24")]]
    print(f"launched 2 stage workers; batched decode of B={len(PROMPTS)} over the wire")

    socks = []
    try:
        tok = AutoTokenizer.from_pretrained(MODEL)
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()

        refs = [seq_ref(model, tok(p, return_tensors="pt").input_ids, N_NEW) for p in PROMPTS]

        for port in (p0, p1):
            wait_listen(port)
        socks = [socket.create_connection(("127.0.0.1", port)) for port in (p0, p1)]

        enc = tok(PROMPTS, return_tensors="pt", padding=True)
        lens = enc.attention_mask.sum(-1).tolist()
        print(f"  ragged lengths {lens} -> padded {enc.input_ids.shape[1]}")
        t0 = time.time()
        batched = batch_decode_socket(model, socks, key, enc.input_ids, enc.attention_mask, N_NEW)
        dt = time.time() - t0

        allok = True
        for b in range(len(PROMPTS)):
            ok = batched[b] == refs[b]
            allok = allok and ok
            print(f"  [{'OK ' if ok else 'BAD'}] seq {b}: {tok.decode(batched[b])!r}")
            if not ok:
                print(f"        ref: {tok.decode(refs[b])!r}")
        assert allok, "batched-over-socket diverged from sequential reference"
        print(f"B2 PASSED — batched decode over the wire is token-identical to sequential "
              f"({N_NEW} tokens, {dt:.1f}s)")
    finally:
        for s in socks:
            try:
                s.close()
            except OSError:
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
