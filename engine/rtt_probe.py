"""
rtt_probe.py — measure coordinator→node round-trip latency for topology-aware routing.

A TCP connect costs one network round-trip (SYN/SYN-ACK), so *timing a connect* to a
node's stage port is a cheap, protocol-agnostic RTT estimate — no ping protocol, works
against any listener, sends no data. The active prober periodically probes every known
node and feeds `registry.set_rtt()`, so proximity routing (topology.route_by_latency)
ranks holders by MEASURED latency rather than the region-distance bootstrap estimate.

Pure stdlib (socket/threading): the timing + one probe-pass logic is unit-testable over
loopback without GPUs (see tests/test_rtt_probe.py).
"""
from __future__ import annotations

import socket
import threading
import time
from typing import Optional

from engine.log import make_logger

log = make_logger("rtt")


def probe_rtt(host: str, port: int, timeout: float = 3.0) -> Optional[float]:
    """Round-trip time (ms) to (host, port) via a timed TCP connect, or None if the
    node is unreachable (refused / timed out). One handshake ≈ one network RTT."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        t0 = time.perf_counter()
        s.connect((host, port))
        return (time.perf_counter() - t0) * 1000.0
    except OSError:
        return None
    finally:
        try:
            s.close()
        except OSError:
            pass


def _probe_once(registry, timeout: float = 3.0) -> int:
    """One probe pass: snapshot dialable node endpoints (under the registry lock),
    probe each UNLOCKED (slow network I/O must not block the control plane), then
    record RTTs (under the lock). Returns how many nodes were successfully probed.
    Unreachable nodes are skipped — their last RTT stays (liveness is the reaper's job)."""
    try:
        targets = registry.node_endpoints()
    except Exception as e:
        log("WARN", "prober snapshot failed", error=str(e))
        return 0
    ok = 0
    for nid, ep in targets:
        try:
            host, port = ep[0], int(ep[1])
        except (TypeError, ValueError, IndexError):
            continue                      # relay / non-dialable endpoint — skip
        ms = probe_rtt(host, port, timeout)
        if ms is not None:
            registry.set_rtt(nid, ms)
            ok += 1
    return ok


def start_rtt_prober(registry, interval: float = 30.0, timeout: float = 3.0) -> threading.Thread:
    """Daemon thread that runs `_probe_once` every `interval` seconds, so measured RTTs
    stay fresh as the network and routes change."""
    def loop():
        while True:
            time.sleep(interval)
            try:
                _probe_once(registry, timeout)
            except Exception as e:      # the prober must never die
                log("WARN", "prober error", error=str(e))

    th = threading.Thread(target=loop, daemon=True)
    th.start()
    log("INFO", "rtt prober started", interval=interval, timeout=timeout)
    return th
