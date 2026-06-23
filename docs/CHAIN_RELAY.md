# Chain Relay — Design

**Status:** design / in progress (pure-logic core built + tested; GPU/mesh wiring staged)
**Lever:** Front 1+3 of `SPEED_ROADMAP.md` — the structural fix for distributed single-stream
latency. **Gated behind `CIRCUIT_CHAIN=1`; default off = the star path, byte-identical.**

---

## Problem

`coordinator._relay_dynamic` is a **star**: for each slot it dials the holder, sends the
activation, reads the result *back to the coordinator*, then dials the next slot. The
activation returns to the coordinator between **every** layer group:

```
T_star = Σ_i 2·RTT(coordinator, holder_i)        # N coordinator round-trips, serial
```

For distributed nodes that's the worst shape — every hop pays the coordinator↔node distance.
If the workers are network-close to each other but far from the coordinator (the common case:
a regional cluster + a remote entry point), the star wastes that adjacency entirely.

## Fix: forward node→node, coordinator only at entry + exit

```
coordinator → holder(slot0) → holder(slot1) → … → holder(slotN) → (result home)
```

The coordinator sends the activation to the **head** holder with the rest of the route
attached. Each node computes its layers, then **forwards** the activation to its next hop
(carrying the shortened route). Two ways the final result gets home:

- **v1 — nested return (this design):** each node forwards to its next hop and *awaits* the
  response on that connection, then returns it upstream on the connection it received on. The
  result bubbles back along the chain; the coordinator does a single send→recv to the head.
  Simple (pure request/response nesting, no new coordinator listener), correct, reuses the
  existing per-conn handler threads.

  ```
  T_chain_v1 = 2·[ RTT(coord,h0) + Σ_i RTT(h_i, h_{i+1}) ]
  ```
  For a regional cluster (h-h hops small) + remote coordinator, ≈ `2·RTT(coord, cluster)` —
  vs the star's `2·N·RTT(coord, cluster)`. **~N× better.**

- **v2 — direct return (optimization, later):** the final node dials the coordinator's
  advertised address and delivers the result directly (no backward bubble), saving the return
  inter-node hops and not blocking intermediate nodes. Needs a coordinator inbound RESULT
  listener + `(session, pos)` correlation. `T_chain_v2 = RTT(coord,h0) + Σ RTT(h_i,h_{i+1}) +
  RTT(h_N, coord)`. Build after v1 proves out.

## KV affinity (unchanged, and that's the point)

The route is already **pinned per session** (`coordinator._session_routes` → warm KV). The
chain uses that same pinned order; each holder still owns its session KV for its layers. The
chain changes only the **data path**, not KV placement — so output is byte-identical to the
star (same nodes, same order, same KV), just routed better. That's why it's exact, not
approximate.

## Wire: `CHAIN_ACTIVATION`

A new message type carries the downstream route ahead of the activation tensor:

```
payload = encode_route(downstream) ++ pack_activation(session, pos, hidden)
encode_route = n_hops:u8, then per hop:  host_len:u8, host, port:u16, key:32
```

`engine/chain.py` owns `encode_route` / `decode_route` / `pop_next` /
`chain_head_and_route` — **pure stdlib, GPU-free, unit-tested** (`tests/test_chain.py`). The
node decodes the route header, pops its next hop, re-encodes the remainder, and forwards.

## Node logic (`stage_worker._handle`, `CHAIN_ACTIVATION` branch)

```
route, rest = decode_route(payload)
session, pos, hidden = unpack_activation(rest)
out = stage.forward(hidden, ...)                  # this node's layers (KV as today)
nxt, remaining = pop_next(route)
if nxt is None:                                   # tail: return result upstream
    write_frame(conn, my_key, RESULT, pack_activation(session, pos, out))
else:                                             # forward + relay the response back (v1)
    fsock = conn_to(nxt.host, nxt.port)           # reuse a pooled per-(session,next) conn
    write_frame(fsock, nxt.key, CHAIN_ACTIVATION, encode_route(remaining) ++ pack_activation(session,pos,out))
    _, back = read_frame_keyed(fsock, nxt.key)    # await downstream RESULT
    write_frame(conn, my_key, RESULT, back)       # bubble it upstream unchanged
```

## Coordinator (`_relay_dynamic`, chain mode)

```
head, downstream = chain_head_and_route(route)    # route = pinned [Node per slot]
sock = self._conn_for(head)
write_frame(sock, head.wire_key, CHAIN_ACTIVATION, encode_route(downstream) ++ pack_activation(session,pos,hidden))
_, payload = read_frame_keyed(sock, head.wire_key)   # the bubbled-back final hidden
return unpack_activation(payload).hidden
```

One coordinator round-trip instead of N.

## Failover

A node's forward-to-next can fail (next hop dead/slow). It sends `ERROR` upstream; the error
bubbles to the coordinator, which marks the broken holder SUSPECT, drops the session pin, and
re-prefills on a fresh route — exactly the existing `_relay_with_failover` re-prefill path,
generalized from "a hop I dialed failed" to "a hop downstream of me failed."

## NAT

Chain forwarding means nodes dial **each other**, not just the coordinator. v1 requires
`reachability="public"` for non-tail nodes (RunPod-proxied nodes qualify — they advertise a
public host:port). NAT'd nodes (`reachability="relay"`) either sit at the **tail** (only their
predecessor dials them — still needs reachability) or route via the existing relay path. Full
NAT/relay chaining is v2; until then the topology should order public nodes into the chain and
fall back to the star when a chain can't be formed.

## Security note (key sharing)

In v1 the coordinator includes each downstream node's `wire_key` in the route, so a node can
encrypt to its successor. This means a node learns its **successor's** key → it could, in
principle, impersonate the coordinator to that successor. Acceptable on a **permissioned**
mesh (vetted operators). For the open network, issue **per-session ephemeral forwarding keys**
(coordinator generates a session chain key, hands each adjacent pair their shared leg key) so
no node holds another's long-lived key. v2 security item; until then run chain mode permissioned.

## Build order

1. **`engine/chain.py` + `tests/test_chain.py`** — route encode/decode + head/next logic
   (pure, GPU-free). ✅ done.
2. `wire.CHAIN_ACTIVATION` constant + a pooled `conn_to(host,port)` on the stage worker.
3. `stage_worker._handle` CHAIN_ACTIVATION branch (forward + nested return).
4. `coordinator._relay_dynamic` chain mode behind `CIRCUIT_CHAIN=1` + `chain_head_and_route`.
5. Validate on a 72B mesh (output identical to star; measure tok/s vs the 1.5 baseline).
6. v2: direct return + per-session forwarding keys + NAT/relay chaining.
