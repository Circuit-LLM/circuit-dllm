"""
scheduler.py — Win B / B3: a batching scheduler in front of the coordinator.

Requests arrive independently (different times, different lengths). The scheduler
collects up to `max_batch` of them into one batch and runs a single batched decode
(Coordinator.generate_batch_stream — one batched forward per step instead of B
separate forwards), streaming each request's tokens back to its own queue. That's the
intra-step batching throughput win made usable by real, concurrent traffic.

This is STATIC batching: a batch runs to completion before the next forms (a request
may wait up to the slowest one in its batch). Dynamic admit/evict — continuous
batching, where a finished slot is immediately refilled — is the next step (B3b); it
improves utilization but rides on exactly this machinery + the batched primitive.
"""

from __future__ import annotations

import queue
import threading
import time

from engine.log import make_logger


class BatchScheduler:
    DONE = object()        # sentinel pushed to a request's queue when it finishes

    def __init__(self, coord, max_batch: int = 8, max_wait: float = 0.02):
        self.coord = coord
        self.max_batch = max_batch
        self.max_wait = max_wait          # how long to let a batch fill before running
        self.log = make_logger("sched")
        self._q: "queue.Queue" = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit(self, prompt: str, max_tokens: int = 64) -> "queue.Queue":
        """Enqueue a request. Returns a Queue the caller drains: token ids, then the
        DONE sentinel (or an Exception instance if the batch failed)."""
        out: "queue.Queue" = queue.Queue()
        self._q.put((prompt, int(max_tokens), out))
        return out

    def _collect(self):
        """Block for the first request, then gather more (up to max_batch) for a short
        window so requests arriving close together share a batch."""
        batch = [self._q.get()]
        deadline = time.time() + self.max_wait
        while len(batch) < self.max_batch:
            timeout = deadline - time.time()
            if timeout <= 0:
                break
            try:
                batch.append(self._q.get(timeout=timeout))
            except queue.Empty:
                break
        return batch

    def _loop(self):
        while not self._stop.is_set():
            batch = self._collect()
            prompts = [p for p, _, _ in batch]
            caps = [m for _, m, _ in batch]
            outs = [o for _, _, o in batch]
            self.log("INFO", "batch", size=len(batch), caps=f"{min(caps)}-{max(caps)}")
            try:
                for b, tok in self.coord.generate_batch_stream(prompts, caps):
                    outs[b].put(tok)
            except Exception as e:   # noqa: BLE001 — surface to callers, keep serving
                self.log("WARN", "batch failed", error=str(e))
                for o in outs:
                    o.put(e)
            for o in outs:
                o.put(self.DONE)

    def stop(self):
        self._stop.set()
