"""
test_mesh_api.py — the LIVE API server in mesh mode, end to end.

Boots engine.api with CIRCUIT_MESH=1 (the gated mesh control plane). Two stage
workers JOIN over the control channel; then a real /v1/chat/completions request is
served, routed through the joined nodes. Proves the control server is correctly
wired into the inference server: API up -> nodes join via /register -> the OpenAI
endpoint answers via the mesh. With CIRCUIT_MESH unset the live path is untouched.

Run on CPU:  python3 -m tests.test_mesh_api
"""

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

MODEL = os.environ.get("CIRCUIT_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
FP = "test-fp"


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def http_json(url, obj=None, timeout=10):
    if obj is None:
        req = urllib.request.Request(url)
    else:
        req = urllib.request.Request(url, data=json.dumps(obj).encode(),
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def wait_http(url, timeout=120):
    end = time.time() + timeout
    while time.time() < end:
        try:
            return http_json(url, timeout=3)
        except (urllib.error.URLError, OSError):
            time.sleep(1)
    raise RuntimeError(f"{url} never came up")


def main():
    key = os.urandom(32).hex()
    api_port, ctrl_port = free_port(), free_port()
    env = {**os.environ,
           "CIRCUIT_MESH": "1", "CIRCUIT_MODEL": MODEL, "CIRCUIT_DEVICE": "cpu",
           "CIRCUIT_KEY": key,
           "CIRCUIT_API_PORT": str(api_port), "CIRCUIT_API_HOST": "127.0.0.1",
           "CIRCUIT_CONTROL_PORT": str(ctrl_port), "CIRCUIT_CONTROL_HOST": "127.0.0.1",
           "CIRCUIT_MESH_LAYERS": "24", "CIRCUIT_MESH_STAGES": "2", "CIRCUIT_MESH_FP": FP,
           "CIRCUIT_MESH_ALLOWLIST": "node-a,node-b", "CIRCUIT_REAP_INTERVAL": "2"}
    # the coordinator co-locates no layers here (CIRCUIT_LOCAL_LAYERS unset) → the
    # mesh covers all 24 layers across 2 slots.
    env.pop("CIRCUIT_LOCAL_LAYERS", None)
    env.pop("CIRCUIT_STAGES", None)
    env.pop("CIRCUIT_DRAFT", None)

    procs = [subprocess.Popen([sys.executable, "-m", "engine.api"], cwd=REPO, env=env)]
    base = f"http://127.0.0.1:{api_port}"
    ctrl = f"http://127.0.0.1:{ctrl_port}"
    try:
        wait_http(base + "/health")
        _, h = wait_http(ctrl + "/health")
        assert h.get("coverage_ok") is False, "no nodes yet → coverage gap expected"
        print("API + control channel up; mesh empty (coverage gap) as expected")

        for nid in ("node-a", "node-b"):
            procs.append(subprocess.Popen(
                [sys.executable, "-m", "engine.stage_worker", "--port", str(free_port()),
                 "--model", MODEL, "--device", "cpu", "--host", "127.0.0.1",
                 "--control-url", ctrl, "--node-id", nid, "--capacity-layers", "12",
                 "--model-fp", FP, "--advertise-host", "127.0.0.1", "--hb-interval", "2"],
                cwd=REPO))
        print("launched 2 nodes joining the live API's control channel")

        for _ in range(180):
            _, h = http_json(ctrl + "/health")
            if h.get("coverage_ok"):
                break
            time.sleep(1)
        assert h.get("coverage_ok"), "mesh never reached full coverage"
        _, w = http_json(base + "/v1/workers")
        snap = w["workers"]
        holders = [h["node_id"] for s in snap["slots"] for h in s["holders"]]
        print(f"  mesh covered; /v1/workers holders = {holders}")
        assert snap["coverage_ok"] and set(holders) == {"node-a", "node-b"}, \
            f"/v1/workers should reflect the 2 joined nodes — got {holders}"

        # a real chat completion, routed through the joined nodes
        code, resp = http_json(base + "/v1/chat/completions",
                               {"model": MODEL, "max_tokens": 16, "stream": False,
                                "messages": [{"role": "user",
                                              "content": "Name the capital of France in one word."}]},
                               timeout=180)
        assert code == 200, resp
        content = resp["choices"][0]["message"]["content"]
        print(f"  completion: {content!r}")
        assert content.strip(), "empty completion from the mesh"
        assert "Paris" in content, f"expected a coherent answer mentioning Paris, got {content!r}"
        print("MESH-API E2E PASSED — API in mesh mode, nodes joined via the control "
              "channel, inference served through the mesh")
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
