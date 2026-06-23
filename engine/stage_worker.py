"""
stage_worker.py — a pipeline stage as a standalone process.

Loads a model, holds layers [start, end), listens on a TCP socket, and serves
the coordinator: each ACTIVATION frame -> run this stage's layers (with per-
session KV) -> return the updated hidden state as a RESULT frame. KV_CTRL frames
reset or roll back a session's cache (for speculative decode).

Run:
  python3 -m engine.stage_worker --port 5001 --layers 0:12 \
      --model Qwen/Qwen2.5-0.5B-Instruct --key <64hex> [--device cpu]

The first frame that decrypts correctly authenticates the peer (shared key);
a bad key raises WireError and the connection is dropped.
"""

from __future__ import annotations

import argparse
import os
import socket
import struct
import threading

import torch

from engine import wire, chain
from engine.tensors import pack_activation, unpack_activation, unpack_batch_activation
from engine.stage import stage_for_range
from engine.kv import StageKV
from engine.model import load_model
from engine.log import make_logger

# KV_CTRL payload: [session u32][op u8][arg u32]
KVOP_RESET = 1
KVOP_TRUNCATE = 2
KVOP_FREE = 3       # drop the session/batch entry entirely (free its KV; Win B batches)


def _position_ids(seq_len: int, start_pos: int, device):
    return (torch.arange(seq_len, device=device) + start_pos).unsqueeze(0)


def _set_keepalive(conn):
    """Enable TCP keepalive so the kernel eventually tears down a vanished peer
    instead of leaving its handler blocked on a zombie socket (which is how a
    coordinator restart wedged the worker before)."""
    try:
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        for name, val in (("TCP_KEEPIDLE", 30), ("TCP_KEEPINTVL", 10), ("TCP_KEEPCNT", 3)):
            opt = getattr(socket, name, None)
            if opt is not None:
                conn.setsockopt(socket.IPPROTO_TCP, opt, val)
    except OSError:
        pass


# Read timeout on an outbound chain-forward socket: it waits for the WHOLE downstream
# chain to compute + return, so it must be generous (slow remote GPUs). Tunable.
_FWD_READ_TIMEOUT = float(os.environ.get("CIRCUIT_FWD_READ_TIMEOUT", "180"))


def _fwd_conn(pool, host, port):
    """A pooled outbound TCP connection to the next chain hop, reused across tokens — a
    fresh handshake per token would re-add a network round-trip and defeat the chain. On
    a wire/socket error the caller drops it from the pool so the next token reconnects."""
    s = pool.get((host, port))
    if s is None:
        s = socket.create_connection((host, port), timeout=10)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s.settimeout(_FWD_READ_TIMEOUT)
        _set_keepalive(s)
        pool[(host, port)] = s
    return s


def _serve_conn(conn, addr, key, stage, config, device, compute_lock, log):
    """Serve one peer connection (its own session/KV space) until it closes."""
    log("INFO", "peer connected", addr=f"{addr[0]}:{addr[1]}")
    sessions: dict[int, StageKV] = {}
    fwd_conns: dict = {}             # (host,port) -> outbound socket for chain forwarding
    try:
        _handle(conn, key, stage, config, device, sessions, log, compute_lock, fwd_conns)
    except (wire.WireError, OSError) as e:
        log("WARN", "connection dropped", reason=str(e))
    finally:
        try:
            conn.close()
        except OSError:
            pass
        for s in fwd_conns.values():      # close the chain-forward sockets this conn opened
            try:
                s.close()
            except OSError:
                pass


def serve(port: int, start: int, end: int, model_id: str, key: bytes,
          device: str = "cpu", host: str = "0.0.0.0",
          prune: bool = False, keep_head: bool = False,
          shard: bool = False, other_device: str = "cpu", quant: str = "",
          on_listening=None):
    log = make_logger(f"stage[{start}:{end}]")
    log("INFO", "loading model", model=model_id, device=device, shard=shard, quant=quant or "fp16")
    if shard:
        # load ONLY this stage's layers into VRAM (models too big to load whole)
        gpu = "cuda:0" if device == "cuda" else device
        if quant == "bnb":
            # 4-bit bitsandbytes: the only quant format that shard-loads (AWQ can't)
            from engine.model import load_model_shard_bnb
            model = load_model_shard_bnb(model_id, start, end, keep_head=keep_head, device=gpu)
            log("INFO", "bnb-shard-loaded owned layers (4bit)", keep_head=keep_head)
        else:
            from engine.model import load_model_shard
            model = load_model_shard(model_id, start, end, keep_head=keep_head,
                                     device=gpu, other_device=other_device)
            log("INFO", "shard-loaded owned layers", keep_head=keep_head, other=other_device)
    else:
        model = load_model(model_id, device=device)
        if prune:
            from engine.model import prune_to_layers
            prune_to_layers(model, start, end, keep_head=keep_head)
            log("INFO", "pruned to owned layers", keep_head=keep_head)
    stage = stage_for_range(model, start, end)
    config = model.config
    compute_lock = threading.Lock()   # serialize model use across peer threads

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(16)
    log("INFO", "listening", port=port)
    if on_listening is not None:       # control-client mode flips JOINING -> READY here
        on_listening()

    # One handler thread per peer. A stalled or half-open peer (e.g. the
    # coordinator pod restarting and its old socket lingering through the RunPod
    # proxy) can no longer wedge the worker: the prior single-connection accept
    # loop blocked forever on the dead socket and refused every new connection.
    # Each peer thread keeps its own session/KV space; threads are daemons.
    while True:
        conn, addr = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # small frames: no Nagle
        _set_keepalive(conn)
        threading.Thread(target=_serve_conn,
                         args=(conn, addr, key, stage, config, device, compute_lock, log),
                         daemon=True).start()


@torch.no_grad()
def _handle(conn, key, stage, config, device, sessions, log, compute_lock, fwd_conns=None):
    if fwd_conns is None:
        fwd_conns = {}
    while True:
        try:
            mt, payload = wire.read_frame_keyed(conn, key)
        except wire.WireError as e:
            if "closed" in str(e):
                return
            raise

        if mt == wire.ACTIVATION:
            session, pos, _flags, hidden = unpack_activation(payload)
            hidden = hidden.to(device)
            cache = sessions.get(session)
            if cache is None or pos == 0:           # pos==0 begins a new sequence
                cache = StageKV(config)
                sessions[session] = cache
            position_ids = _position_ids(hidden.shape[1], pos, device)
            with compute_lock:                      # one model fwd at a time across peers
                out = stage.forward(hidden, position_ids, past_key_values=cache, use_cache=True)
            wire.write_frame(conn, key, wire.RESULT, pack_activation(session, pos, out.cpu()))

        elif mt == wire.BATCH_ACTIVATION:
            # Win B: a batched hidden [B,T,D] with its own per-row position_ids and 2D
            # padding mask — the coordinator computed them, the stage just applies them.
            batch_id, pos, _flags, hidden, position_ids, attn = unpack_batch_activation(payload)
            hidden = hidden.to(device)
            position_ids = position_ids.to(device).long()
            attn = attn.to(device)
            cache = sessions.get(batch_id)
            if cache is None or pos == 0:           # pos==0 begins a fresh batch
                cache = StageKV(config)
                sessions[batch_id] = cache
            with compute_lock:
                out = stage.forward(hidden, position_ids, past_key_values=cache,
                                    use_cache=True, attention_mask=attn)
            wire.write_frame(conn, key, wire.RESULT, pack_activation(batch_id, pos, out.cpu()))

        elif mt == wire.CHAIN_ACTIVATION:
            # Chain relay: compute my layers, then FORWARD the activation to the next hop
            # (carrying the shortened route) and bubble its reply back upstream — instead
            # of returning to the coordinator. The coordinator only touches entry + exit.
            # KV is maintained exactly as in the ACTIVATION path (pos==0 resets), so output
            # is byte-identical to the star. See docs/CHAIN_RELAY.md.
            route, rest = chain.decode_route(payload)
            session, pos, _flags, hidden = unpack_activation(rest)
            hidden = hidden.to(device)
            cache = sessions.get(session)
            # pos==0 resets this session's KV — and this IS the chain failover recovery: on a
            # mid-chain hop failure the coordinator re-prefills the session at pos==0 on a
            # fresh route, which lands here and clears any half-advanced KV. (In chain mode a
            # KV_CTRL reset reaches only the head; non-head nodes rely on this pos==0 reset.)
            if cache is None or pos == 0:
                cache = StageKV(config)
                sessions[session] = cache
            position_ids = _position_ids(hidden.shape[1], pos, device)
            with compute_lock:
                out = stage.forward(hidden, position_ids, past_key_values=cache, use_cache=True)
            nxt, remaining = chain.pop_next(route)
            if nxt is None:                                  # tail: return result upstream
                wire.write_frame(conn, key, wire.RESULT, pack_activation(session, pos, out.cpu()))
            else:                                            # forward + bubble reply back
                nhost, nport, nkey = nxt
                try:
                    fsock = _fwd_conn(fwd_conns, nhost, nport)
                    wire.write_frame(fsock, nkey, wire.CHAIN_ACTIVATION,
                                     chain.encode_route(remaining)
                                     + pack_activation(session, pos, out.cpu()))
                    mt2, payload2 = wire.read_frame_keyed(fsock, nkey)
                    wire.write_frame(conn, key, mt2, payload2)   # RESULT (or ERROR) up, unchanged
                except (wire.WireError, OSError):
                    s = fwd_conns.pop((nhost, nport), None)
                    if s is not None:
                        try:
                            s.close()
                        except OSError:
                            pass
                    # tell the coordinator WHICH hop broke so it suspects the right node
                    wire.write_frame(conn, key, wire.ERROR, f"{nhost}:{nport}".encode())

        elif mt == wire.KV_CTRL:
            session, op, arg = struct.unpack(">IBI", payload[:9])
            if op == KVOP_FREE:
                sessions.pop(session, None)         # drop the entry; frees its KV
            else:
                cache = sessions.get(session)
                if cache is not None:
                    if op == KVOP_RESET:
                        cache.reset()
                    elif op == KVOP_TRUNCATE:
                        cache.truncate_to(arg)

        elif mt == wire.PING:
            wire.write_frame(conn, key, wire.PONG, payload)

        elif mt == wire.BYE:
            return


# ── control-client mode: join a coordinator's mesh over HTTP ──────────────────

def _control_post(url, obj, timeout=10):
    import json as _json
    import urllib.request as _u
    data = _json.dumps(obj).encode()
    req = _u.Request(url, data=data, headers={"Content-Type": "application/json"})
    with _u.urlopen(req, timeout=timeout) as r:
        return r.status, _json.loads(r.read())


def _resolve_node_identity(a):
    """Resolve this node's ed25519 keypair for SIGNED registration and return
    (private_key, node_id_hex). Priority: --node-key hex, else a persisted key file
    (stable identity across restarts), else generate one and save it. node_id is the
    public key hex — the coordinator's ed25519 verifier proves the registrant holds
    the matching private key, so a node can't impersonate another's id."""
    import os
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives import serialization
    RAW, RAWPRIV = serialization.Encoding.Raw, serialization.PrivateFormat.Raw
    if a.node_key:
        sk = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(a.node_key.strip()))
    else:
        kf = a.node_key_file or os.environ.get("CIRCUIT_NODE_KEY_FILE") or "/workspace/node_key.hex"
        if os.path.exists(kf):
            sk = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(open(kf).read().strip()))
        else:
            sk = Ed25519PrivateKey.generate()
            try:
                d = os.path.dirname(kf)
                if d:
                    os.makedirs(d, mode=0o700, exist_ok=True)
                raw = sk.private_bytes(RAW, RAWPRIV, serialization.NoEncryption())
                # Create the key file 0600 atomically (O_EXCL) — no world-readable
                # window between create and chmod for this private-key credential.
                fd = os.open(kf, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(fd, "w") as f:
                    f.write(raw.hex())
            except OSError:
                pass
    node_id = sk.public_key().public_bytes(RAW, serialization.PublicFormat.Raw).hex()
    return sk, node_id


def run_control_client(a):
    """Join a coordinator's mesh: register over the control channel, receive an
    assigned layer range + a per-node key, then serve those layers with that key
    while heartbeating. Heartbeats start immediately (so the JOINING node isn't
    reaped during the weight load); /ready flips it to READY once listening; /drain
    on exit."""
    clog = make_logger("node")
    base = a.control_url.rstrip("/")
    advertise = a.advertise_host or ("127.0.0.1" if a.host in ("0.0.0.0", "") else a.host)
    # the endpoint the COORDINATOR dials may differ from the local bind (NAT / a
    # RunPod TCP proxy maps an external port -> the internal --port). Advertise the
    # externally-reachable port when given, else the bind port.
    advertise_port = a.advertise_port or a.port

    import json as _json, time as _time
    reg = {
        "node_id": a.node_id,
        "endpoint": [advertise, advertise_port],
        "capacity_layers": a.capacity_layers,
        "model_fp": a.model_fp,
        "reachability": "public",
        "region": a.region or "",          # coarse geo label for proximity routing (signed)
        "payout_wallet": a.payout_wallet or "",
        "ts": int(_time.time()),
    }
    import urllib.error as _ue

    def _sign_register():
        # Sign the canonical body (minus sig) so the coordinator's ed25519 verifier can
        # prove we hold the private key behind node_id. Refresh ts + re-sign on each
        # attempt so a long retry window can't drift outside the replay tolerance.
        reg["ts"] = int(_time.time())
        reg.pop("sig", None)
        if getattr(a, "_signing_key", None) is not None:
            msg = _json.dumps(reg, sort_keys=True, separators=(",", ":")).encode()
            reg["sig"] = a._signing_key.sign(msg).hex()

    def _do_register(prefer_range=None, timeout=2400.0):
        """POST /register (signed), retrying while the coordinator is still coming up — its
        control plane may not be listening yet when this node finishes its own (slow) model
        download, and a node should be able to JOIN a mesh whose coordinator starts later. A
        connection error / 5xx is transient (keep trying); a 4xx is a real rejection (bad
        signature, model_fp mismatch, no capacity) — fail fast. `prefer_range=(start,end)`
        asks for an already-loaded slot back (used on re-register). Returns (start, end, key)."""
        if prefer_range is not None:
            reg["loaded_layers"] = list(prefer_range)
        else:
            reg.pop("loaded_layers", None)
        deadline = _time.time() + float(timeout)
        attempt = 0
        while True:
            attempt += 1
            _sign_register()
            try:
                _code, resp = _control_post(base + "/register", reg)
                break
            except _ue.HTTPError as e:
                if 400 <= e.code < 500:
                    body = ""
                    try: body = e.read().decode()[:200]
                    except Exception: pass
                    raise SystemExit(f"register rejected ({e.code}): {body}")
                last = f"http {e.code}"          # 5xx — transient
            except Exception as e:
                last = str(e)                    # connection refused etc — coordinator not up yet
            if _time.time() >= deadline:
                raise SystemExit(f"register failed after {attempt} attempts: {last}")
            clog("INFO", "coordinator not ready, retrying register", attempt=attempt, last=last)
            _time.sleep(10)
        return (resp["assignment"]["start"], resp["assignment"]["end"],
                wire.normalize_key(bytes.fromhex(resp["session_key"])))

    start, end, key = _do_register(timeout=float(getattr(a, "register_timeout", 2400)))
    clog("INFO", "joined mesh", node=a.node_id[:12], layers=f"{start}:{end}")

    stop = threading.Event()

    def _heartbeat():
        while not stop.is_set():
            try:
                _c, _resp = _control_post(base + "/heartbeat", {"node_id": a.node_id}, timeout=5)
                # If the coordinator no longer knows us (it restarted with an empty topology),
                # RE-REGISTER for our already-loaded slot + re-announce READY — serve() keeps
                # running and (same master secret → same derived key) its key stays valid. This
                # is what makes a coordinator restart NOT orphan the nodes.
                if isinstance(_resp, dict) and _resp.get("registered") is False:
                    clog("WARN", "coordinator forgot us — re-registering", layers=f"{start}:{end}")
                    try:
                        s2, e2, _k2 = _do_register(prefer_range=(start, end), timeout=120)
                        if (s2, e2) != (start, end):
                            clog("ERROR", "re-register got a different slot — serving stale layers",
                                 had=f"{start}:{end}", got=f"{s2}:{e2}")
                        _control_post(base + "/ready", {"node_id": a.node_id}, timeout=5)
                        clog("INFO", "re-registered + ready")
                    except SystemExit as e:       # 4xx rejection — don't kill the heartbeat thread
                        clog("ERROR", "re-register rejected", error=str(e))
                    except Exception as e:
                        clog("WARN", "re-register failed", error=str(e))
            except Exception:
                pass
            stop.wait(a.hb_interval)

    threading.Thread(target=_heartbeat, daemon=True).start()

    def _on_listening():
        try:
            _control_post(base + "/ready", {"node_id": a.node_id}, timeout=5)
            clog("INFO", "ready (serving)")
        except Exception as e:
            clog("WARN", "ready post failed", error=str(e))

    try:
        serve(a.port, start, end, a.model, key, device=a.device, host=a.host,
              prune=a.prune, shard=a.shard, other_device=a.other_device, quant=a.quant,
              on_listening=_on_listening)
    finally:
        stop.set()
        try:
            _control_post(base + "/drain", {"node_id": a.node_id}, timeout=5)
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--prune", action="store_true",
                    help="load whole model then free non-owned layers (bnb; fits-one-card models)")
    ap.add_argument("--shard", action="store_true",
                    help="load ONLY owned layers into VRAM (for models too big for one card)")
    ap.add_argument("--other-device", default="cpu",
                    help="where non-owned modules go under --shard (cpu or meta)")
    ap.add_argument("--quant", default="",
                    help="quant format for --shard: 'bnb' (4-bit bitsandbytes, the only "
                         "format that shard-loads) or '' (fp16). AWQ can't shard.")
    ap.add_argument("--keep-head", action="store_true",
                    help="keep embed/norm/lm_head (for the coordinator-colocated stage)")
    # ── static mode: a fixed layer range + the shared cluster key ──
    ap.add_argument("--layers", help="START:END for static mode (e.g. 0:12)")
    ap.add_argument("--key", help="64-char hex cluster key for static mode")
    # ── control-client mode: join a coordinator's mesh; receive layers + a per-node key ──
    ap.add_argument("--control-url", help="join the mesh via this coordinator control URL")
    ap.add_argument("--node-id", help="this node's id (ed25519 pubkey hex); derived from the key if a key is resolved")
    ap.add_argument("--node-key", help="ed25519 private key hex; node-id derives from it (signs /register)")
    ap.add_argument("--node-key-file", help="persist/load the node key here (default /workspace/node_key.hex) for a stable identity across restarts")
    ap.add_argument("--region", default="",
                    help="coarse geo label (e.g. na-east) for proximity routing; signed in /register")
    ap.add_argument("--capacity-layers", type=int, default=999,
                    help="max contiguous layers this node can hold")
    ap.add_argument("--model-fp", default="", help="fingerprint of the model this node loads")
    ap.add_argument("--advertise-host", default="",
                    help="host the coordinator dials to reach this node (defaults to --host)")
    ap.add_argument("--advertise-port", type=int, default=0,
                    help="port the coordinator dials (for a NAT/proxy that maps an "
                         "external port to --port; defaults to --port)")
    ap.add_argument("--payout-wallet", default="", help="where CIRC earnings settle")
    ap.add_argument("--hb-interval", type=float, default=10.0)
    a = ap.parse_args()

    if a.control_url:
        sk, node_id = _resolve_node_identity(a)
        if a.node_id and a.node_id != node_id:
            ap.error(f"--node-id does not match the resolved key's pubkey ({node_id})")
        a.node_id = node_id
        a._signing_key = sk
        run_control_client(a)
        return

    if not a.layers or not a.key:
        ap.error("--layers and --key are required in static mode (or use --control-url)")
    start, end = (int(x) for x in a.layers.split(":"))
    serve(a.port, start, end, a.model, wire.normalize_key(a.key), a.device, a.host,
          prune=a.prune, keep_head=a.keep_head, shard=a.shard, other_device=a.other_device,
          quant=a.quant)


if __name__ == "__main__":
    main()
