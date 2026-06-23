"""
test_rtt_probe.py — the active RTT prober for topology-aware routing.

Covers: TCP-connect timing over loopback, unreachable → None, and one full probe pass
against a fake registry (live endpoint recorded, dead endpoint skipped, relay skipped).

Pure stdlib over loopback, no GPUs:  python3 -m tests.test_rtt_probe
"""

import os
import socket
import sys
import threading

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from engine.rtt_probe import probe_rtt, _probe_once  # noqa: E402


def _listener():
    """A throwaway TCP listener on a free loopback port; accepts + closes."""
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", 0))
    s.listen(8)
    port = s.getsockname()[1]

    def serve():
        while True:
            try:
                c, _ = s.accept()
                c.close()
            except OSError:
                break

    threading.Thread(target=serve, daemon=True).start()
    return s, port


def _free_closed_port():
    """Bind then immediately release a port → a port that will refuse connections."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class _FakeRegistry:
    """Minimal stand-in: just the two methods the prober uses."""
    def __init__(self, targets):
        self._targets = targets
        self.recorded = {}

    def node_endpoints(self):
        return self._targets

    def set_rtt(self, node_id, ms):
        self.recorded[node_id] = ms


def main():
    srv, live_port = _listener()
    closed_port = _free_closed_port()

    # ── timing: a live loopback connect returns a small, non-negative RTT ──────
    ms = probe_rtt("127.0.0.1", live_port, timeout=2.0)
    assert ms is not None and 0.0 <= ms < 1000.0, f"loopback probe should be fast, got {ms}"

    # ── unreachable: refused port → None ──────────────────────────────────────
    assert probe_rtt("127.0.0.1", closed_port, timeout=1.0) is None, "refused → None"

    # ── one probe pass: live recorded, dead skipped, relay skipped ────────────
    reg = _FakeRegistry([
        ("live", ("127.0.0.1", live_port)),
        ("dead", ("127.0.0.1", closed_port)),
        ("relay", ("relay", "abc")),        # non-dialable endpoint → skipped, no crash
    ])
    n = _probe_once(reg, timeout=1.0)
    assert n == 1, f"only the live node should be probed, got {n}"
    assert "live" in reg.recorded and reg.recorded["live"] >= 0.0, "live RTT recorded"
    assert "dead" not in reg.recorded, "unreachable node not recorded"
    assert "relay" not in reg.recorded, "relay endpoint skipped"

    srv.close()
    print("RTT PROBE TESTS PASSED — loopback timing, unreachable → None, probe pass (live/dead/relay)")


if __name__ == "__main__":
    main()
