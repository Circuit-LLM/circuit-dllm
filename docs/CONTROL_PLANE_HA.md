# Control-Plane HA — active + warm standby (the no-chain SPOF fix)

**Status:** BUILT + unit-tested (off-prod); deploy is staged + gated. The pragmatic, **zero-SOL,
zero-VPS** fix for the one risk left after the floating-coordinator cutover: the control plane is a
single process on a single pod (`docs/CUTOVER_FLOATING.md` "KNOWN RISK"). The fully-decentralized
end-state is `docs/ONCHAIN_CONTROL_PLANE.md` (parked, free, on devnet) — **this** is the cheap
interim that removes the SPOF today without spending anything.

## Idea

The control plane is a featherweight registry/router (no GPU, off the per-token path). So run **two
of them** — an **active** + a **warm standby** — as ordinary processes on **two different pods you
already pay for** (e.g. active on `hold0`, standby on `orch1`/`hold1`, deliberately different hosts).
No new machine, no VPS, no SOL. Clients hold the **list** and fail over.

## How clients use it (`engine/route_provider.py` → `ControlEndpoints`)

- **`post_all`** — register / ready / heartbeat fan out to **every** control plane, so the standby is
  always warm (knows every node, READY) and can take over with no cold start.
- **`post`** — per-request RPCs (`/route/*`, `/entry/*`, heartbeat) try the **sticky-preferred**
  control plane, then the rest; the first that answers becomes preferred. Failover is transparent and
  doesn't storm a dead node.

Wired in: orchestrators (`engine/api.py _run_orchestrator`) register/ready/heartbeat to all + route
via the failover provider; holders (`engine/stage_worker.py run_control_client`) register on the
active, **mirror to standbys with `prefer_range`** (their already-loaded slot), and heartbeat all; the
gateway (`circuit-data-api/inference-gateway.js`) fails `acquire_entry` over the `CIRCUIT_CONTROL_URLS`
list. Single-URL `CIRCUIT_CONTROL_URL` still works (back-compat).

## Why two independent control planes stay consistent

- **Layout:** a holder mirrors to the standby passing `prefer_range = its active-assigned slot`; the
  registry honors it → the standby's layout is byte-identical (no assignment drift). *Tested:
  `tests/test_control_failover.py::test_holders_mirror_to_standby_identical_and_routable`.*
- **Wire keys:** derived from the **shared cluster key** (`master_secret`), so both control planes
  hand a holder the *same* key — serving survives a failover without a rekey.
- **Session-id prefix (`orch_index`):** private to each orchestrator (namespaces its session ids,
  not used for routing), so it needn't match across control planes — only be unique per orchestrator,
  which each control plane guarantees.

## Failure modes

- **Active control plane / its pod dies:** clients' next `post` fails over to the standby (already
  warm) → routing continues. Heartbeats keep the standby fresh; a cold standby would repopulate within
  one heartbeat interval via the existing re-register-on-`registered:false` path.
- **Standby dies:** no effect (active serves); it re-warms when it returns.
- **Both die:** `post` raises / gateway falls back to the fixed ENGINE — same as today's single-CP loss.
- **Split state:** impossible to cause wrong output — each holder serves its slot with its derived key
  regardless of which control plane routed; a stale view just means a failed hop → failover.

## Config / deploy (staged, gated)

Set on **every** node + the gateway, active first:
```
CIRCUIT_CONTROL_URLS="http://<active-host:port>,http://<standby-host:port>"
```
1. Launch a 2nd control plane (`CIRCUIT_ROLE=control`, same cluster key + mesh config) on a **different
   pod** from the active.
2. Roll `CIRCUIT_CONTROL_URLS` to orchestrators + holders + the gateway (rolling restart).
3. Verify: kill the active → the standby serves `acquire_entry`/`acquire_route`; bring it back → warm.

## Honest caveat — this fixes the control plane, not holder loss

A control-plane standby keeps the **control plane** alive when its pod dies. But if that pod **also**
holds a layer slot at **replication=1** (today's prod: the active CP rides on `hold0`, which holds
`[0,40)`), that slot has no replica → the mesh still can't serve it until a replacement holder loads.
Full pod-death resilience = control-plane HA **+ `replication ≥ 2`** (more GPU pods = $). This change
delivers the first half for free; the second is a separate cost decision. Best immediate placement:
run the **active control plane on a pod that is NOT a singleton holder**, standby on another host.
