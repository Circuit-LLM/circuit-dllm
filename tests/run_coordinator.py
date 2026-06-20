"""
run_coordinator.py — run a coordinator against (possibly remote) stage workers.

Used for the cross-pod test: the coordinator runs on pod 1 and drives stage
workers that may live on another pod, reached over the encrypted wire.

Env:
  CIRCUIT_MODEL   model id
  CIRCUIT_KEY     64-char hex cluster key (must match the workers)
  CIRCUIT_STAGES  comma list of host:port (in pipeline order)
  CIRCUIT_PROMPT  prompt (optional)
  CIRCUIT_N       tokens to generate (optional, default 20)
  CIRCUIT_DEVICE  cuda | cpu (default cuda)
  CIRCUIT_SPEC    if "1", use generate_speculative
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.coordinator import Coordinator  # noqa: E402

MODEL = os.environ["CIRCUIT_MODEL"]
KEY = bytes.fromhex(os.environ["CIRCUIT_KEY"])
STAGES = [(h, int(p)) for h, p in (a.split(":") for a in os.environ["CIRCUIT_STAGES"].split(","))]
PROMPT = os.environ.get("CIRCUIT_PROMPT", "The capital of France is")
N = int(os.environ.get("CIRCUIT_N", "20"))
DEVICE = os.environ.get("CIRCUIT_DEVICE", "cuda")
SPEC = os.environ.get("CIRCUIT_SPEC") == "1"
# CIRCUIT_LOCAL_LAYERS="0:32" runs those layers in-process (co-located stage 0)
_ll = os.environ.get("CIRCUIT_LOCAL_LAYERS")
LOCAL_LAYERS = tuple(int(x) for x in _ll.split(":")) if _ll else None
DRAFT = os.environ.get("CIRCUIT_DRAFT") or None
SHARD = os.environ.get("CIRCUIT_SHARD") == "1"
OTHER_DEVICE = os.environ.get("CIRCUIT_OTHER_DEVICE", "cpu")


def main():
    print(f"coordinator: model={MODEL} stages={STAGES} device={DEVICE} "
          f"local={LOCAL_LAYERS} shard={SHARD} spec={SPEC}")
    coord = Coordinator(MODEL, STAGES, KEY, device=DEVICE,
                        local_layers=LOCAL_LAYERS, draft_model_id=DRAFT,
                        shard=SHARD, other_device=OTHER_DEVICE)
    t0 = time.time()
    if SPEC:
        text, toks = coord.generate_speculative(PROMPT, N)
    else:
        text, toks = coord.generate(PROMPT, N)
    dt = time.time() - t0
    print(f"CROSSPOD_RESULT tok/s={N/dt:.2f} time={dt:.2f}s")
    print(f"CROSSPOD_TEXT {text!r}")
    coord.close()


if __name__ == "__main__":
    main()
