# Circuit DLLM — NAT relay (home-desktop GPUs)

How a GPU behind a home router joins the mesh **without port-forwarding**. This is the piece
that turns "cloud GPUs only" into "anyone with a desktop GPU." **IMPLEMENTED** (`engine/relay.py`
+ node/coordinator integration; relay server covered by `tests/test_relay.py`). Needs validation
on a real cross-NAT mesh.

## Run it

```bash
# relay server (a small public box near the coordinator):
CIRCUIT_RELAY_TOKEN=<shared-coord-token> python3 -m engine.relay --port 18940

# coordinator: presents the token when dialing through the relay
export CIRCUIT_RELAY_TOKEN=<shared-coord-token>

# home node: just set the relay URL — the one-line installer passes it through
CIRCUIT_RELAY_URL=relay.circuitllm.xyz:18940 CIRCUIT_RELAY_TOKEN=<...> ...
```

A relay node advertises `reachability:"relay"` with the relay's address as its endpoint, so the
coordinator's `_conn_for` dials *through* the relay transparently — the rest of the engine is
unchanged. Knobs: `CIRCUIT_RELAY_URL` (node), `CIRCUIT_RELAY_TOKEN` (node + coordinator),
`CIRCUIT_RELAY_PORT` (server).

---

## The problem

The data plane has the coordinator **dial into** each stage (see `JOIN.md` §0): the coordinator
opens a TCP connection to the node's advertised `host:port` and streams `ACTIVATION` frames.
That works for a cloud GPU (public IP) but **not** for a home desktop: it sits behind NAT, so it
has no inbound-reachable address. Asking users to forward a router port is the #1 onboarding
killer — most won't or can't.

A node can always make **outbound** connections, though. The relay exploits that: the node dials
out to a rendezvous, and the coordinator reaches it *through* that connection. Same model as
ngrok / frp / a TURN server.

## The design (ngrok-style control + on-demand data conns)

```
   home GPU (NAT)                 relay (public)                coordinator
   ──────────────                 ─────────────                 ───────────
   1. CONTROL conn  ───outbound──▶  register node_id, hold
                                                          ◀── 2. "dial node X" (advertise=relay)
                    ◀── 3. "open data conn for session S" ──
   4. DATA conn S   ───outbound──▶  pair with coordinator ◀───── 4. coordinator's conn for S
                       ◀═══════ ChaCha20 frames piped 1:1 ═══════▶
```

1. **Control connection.** On startup the node opens one persistent outbound TCP connection to
   the relay and authenticates with its **ed25519 node key** (the same identity it registers with
   the coordinator — the relay verifies a signed nonce, so a node can't squat another's id). The
   relay keeps `{node_id → control_conn}` and holds the connection open (keepalive/PING).

2. **Advertise the relay.** When the node registers with the coordinator (`/register`), instead of
   its own unreachable IP it advertises **the relay's** `host:port` plus its `node_id` as a routing
   tag. The coordinator's router treats it like any other endpoint — it just happens to be the relay.

3. **On-demand data connection.** When the coordinator opens a stage connection (it may open
   several — one per concurrency slot), it connects to the relay and names the target `node_id` +
   a fresh session tag. The relay signals the node over its control connection: *"open a data
   connection for session S."*

4. **Pair + pipe.** The node opens a new outbound DATA connection tagged S; the relay pairs it
   with the waiting coordinator connection and pipes bytes **1:1** in both directions. No
   multiplexing protocol needed — connection-pairing keeps the relay dumb and the engine's framed
   wire unchanged.

When the coordinator closes a slot, both legs close; idle data conns time out.

## Security — the relay is untrusted plumbing

The data wire is already **ChaCha20-encrypted end-to-end** under the node's per-node key (keyed at
`/register`), and frames are length-delimited. The relay only ever sees **ciphertext** and copies
bytes — it cannot read or forge activations. So a community-run or third-party relay is acceptable;
it can DoS (drop bytes) but not eavesdrop. Control-channel auth (signed nonce) stops node-id
spoofing; per-node rate limits cap abuse.

## Latency

One extra hop: `node → relay → coordinator` instead of `coordinator → node` directly. Put the
relay **near the coordinator** so the added leg is the node's home uplink — which is on the
critical path regardless. Expect home nodes to be the slowest stages; topology-aware routing
(`TOPOLOGY_AWARE_ROUTING.md`) should weight them lighter / place them as a single short stage.

## Status

| Piece | Where | State |
|------|-------|-------|
| Relay server (control + data pairing, ed25519 node auth, keepalive) | `engine/relay.py` | DONE (loopback-tested) |
| Node-side relay dialer (control conn + on-demand data conns) | `engine/stage_worker.py` — `CIRCUIT_RELAY_URL` | DONE |
| Coordinator "dial via relay" | `coordinator._relay_dial` / `_conn_for` (reachability=relay) | DONE |
| Reconnect/backoff (node control) | `_relay_client` | DONE |
| Cross-NAT mesh validation; idle reaping + metrics; HA (multiple relays) | relay + node | TODO |

**Phasing:** ship cloud/public-IP onboarding first (works today, biggest GPUs). Add the relay as
Phase 3 so home desktops join with the same one-line installer — just `CIRCUIT_RELAY_URL` set.
The installer already passes it through.
