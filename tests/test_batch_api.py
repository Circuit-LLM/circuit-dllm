"""
test_batch_api.py — Win B / B3: the LIVE API in batch mode, end to end.

Boots engine.api with CIRCUIT_BATCH=1 (co-located stage 0:12 + a remote stage 12:24),
fires concurrent /v1/chat/completions, and checks each is served coherently — i.e. the
API correctly routes through the BatchScheduler (submit -> batched decode -> per-request
stream). With CIRCUIT_BATCH unset the live path is untouched (separate branch).

Run on CPU:  python3 -m tests.test_batch_api
"""

import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

MODEL = os.environ.get("CIRCUIT_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def http_json(url, obj=None, timeout=120):
    req = (urllib.request.Request(url) if obj is None else
           urllib.request.Request(url, data=json.dumps(obj).encode(),
                                  headers={"Content-Type": "application/json"}))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def wait_http(url, timeout=180):
    end = time.time() + timeout
    while time.time() < end:
        try:
            return http_json(url, timeout=3)
        except (urllib.error.URLError, OSError):
            time.sleep(1)
    raise RuntimeError(f"{url} never came up")


def main():
    key = os.urandom(32).hex()
    wport, aport = free_port(), free_port()
    worker = subprocess.Popen(
        [sys.executable, "-m", "engine.stage_worker", "--port", str(wport),
         "--layers", "12:24", "--model", MODEL, "--key", key, "--device", "cpu"],
        cwd=REPO)
    env = {**os.environ, "CIRCUIT_MODEL": MODEL, "CIRCUIT_DEVICE": "cpu", "CIRCUIT_KEY": key,
           "CIRCUIT_STAGES": f"127.0.0.1:{wport}", "CIRCUIT_LOCAL_LAYERS": "0:12",
           "CIRCUIT_API_PORT": str(aport), "CIRCUIT_API_HOST": "127.0.0.1",
           "CIRCUIT_BATCH": "1", "CIRCUIT_MAX_BATCH": "4"}
    env.pop("CIRCUIT_DRAFT", None)
    env.pop("CIRCUIT_MESH", None)
    api = subprocess.Popen([sys.executable, "-m", "engine.api"], cwd=REPO, env=env)
    procs = [worker, api]
    base = f"http://127.0.0.1:{aport}"

    reqs = [("What is the capital of France? Answer in one word.", "Paris"),
            ("What is two plus two? Answer with a number.", None),
            ("Name the largest planet. One word.", None),
            ("What gas do plants take in? One word.", None)]
    try:
        wait_http(base + "/health")
        print("API up in batch mode; firing 4 concurrent requests")
        results = [None] * len(reqs)

        def fire(i, prompt):
            code, resp = http_json(base + "/v1/chat/completions",
                                   {"model": "x", "max_tokens": 24, "stream": False,
                                    "messages": [{"role": "user", "content": prompt}]})
            results[i] = (code, resp)

        threads = [threading.Thread(target=fire, args=(i, p)) for i, (p, _) in enumerate(reqs)]
        t0 = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        dt = time.time() - t0

        allok = True
        for i, (prompt, want) in enumerate(reqs):
            code, resp = results[i]
            content = resp.get("choices", [{}])[0].get("message", {}).get("content", "") if code == 200 else ""
            ok = code == 200 and bool(content.strip()) and (want is None or want in content)
            allok = allok and ok
            print(f"  [{'OK ' if ok else 'BAD'}] req {i} (code {code}): {content[:60]!r}")
        assert allok, "a batched API request failed or returned wrong/empty content"
        print(f"B3 API PASSED — API in batch mode served 4 concurrent requests via the "
              f"scheduler ({dt:.1f}s)")
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()


if __name__ == "__main__":
    main()
