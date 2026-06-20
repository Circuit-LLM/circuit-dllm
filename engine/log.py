"""log.py — tiny structured logger (stderr, one line per event)."""

from __future__ import annotations

import sys
import time


def make_logger(name: str):
    def log(level: str, msg: str, **kw):
        ts = time.strftime("%H:%M:%S")
        extra = " ".join(f"{k}={v}" for k, v in kw.items())
        line = f"{ts} [{name}] {level:5s} {msg}"
        if extra:
            line += "  " + extra
        print(line, file=sys.stderr, flush=True)
    return log
