#!/usr/bin/env python3
"""Staging floating-coordinator mesh across 4 RunPod pods (scattered, real RTT) + a throughput sweep.

Roles:  pod0 = standalone control plane (CPU process) + holder replica #1
        pod1 = holder replica #2                       (replication=2 -> 2 lanes)
        pod2, pod3 = head-only orchestrators
Cross-pod wiring uses each pod's EXTERNAL RunPod-proxy host:port (advertise), so an orchestrator on
one pod can dial a holder on another.  Model: 1.5B target + 0.5B draft (fast holders -> the
orchestrator's draft+head is what saturates = the funnel under test).

  python3 staging.py validate   # control + 1 holder + 1 orchestrator, cross-pod greedy completion
  python3 staging.py sweep       # 2 holders + 2 orchestrators, streaming tok/s: 1 orch vs 2 orch
"""
import json, os, subprocess, sys, time, urllib.request, concurrent.futures as cf

# Pods come from prov-pod.py CLEAN_POD logs named stg0..stg3.log in STAGING_LOG_DIR.
SCR = os.environ.get("STAGING_LOG_DIR", ".")
KEY = os.environ.get("STAGING_SSH_KEY", os.path.expanduser("~/.ssh/id_ed25519"))
SSHOPT = ["-o","StrictHostKeyChecking=no","-o","UserKnownHostsFile=/dev/null","-o","ConnectTimeout=15"]
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DRAFT = "Qwen/Qwen2.5-0.5B-Instruct"
LAYERS = 28                       # Qwen2.5-1.5B layer count (1 slot covers [0,28))
CKEY = "11" * 32                  # shared cluster/master secret for the staging mesh
CONTROL_PORT, HOLDER_PORT, ORCH_PORT = 18932, 19320, 18931
LOCAL_CONTROL = f"http://127.0.0.1:18932"   # a node CO-LOCATED with the control plane reaches it via
                                            # localhost (RunPod has no hairpin NAT to a pod's own external IP)
PROMPT = "Explain in one paragraph why the sky is blue."


def secret(n):
    return subprocess.check_output(["/home/watchtower/.openclaw/credentials/infisical-get.sh", n]).decode().strip()
RPK = secret("RUNPOD_API_KEY")


def gql(q):
    req = urllib.request.Request("https://api.runpod.io/graphql?api_key=" + RPK,
        data=json.dumps({"query": q}).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "curl/8.5.0"})
    return json.load(urllib.request.urlopen(req, timeout=20))


def load_pods():
    """Read the 4 stgN CLEAN_POD logs + fetch each pod's external port map (privatePort->publicPort)."""
    pods = []
    for n in range(4):
        line = [l for l in open(f"{SCR}/stg{n}.log") if l.startswith("CLEAN_POD")]
        if not line:
            sys.exit(f"stg{n}: no CLEAN_POD yet")
        d = json.loads(line[-1][len("CLEAN_POD "):])
        pid, ip, sshp = d["id"], d["ip"], d["ssh"]
        ports = (gql('query{pod(input:{podId:"%s"}){runtime{ports{ip privatePort publicPort type}}}}' % pid)
                 ["data"]["pod"]["runtime"]["ports"])
        ext = {p["privatePort"]: (p["ip"], p["publicPort"]) for p in ports}
        pods.append({"n": n, "id": pid, "ip": ip, "ssh": sshp, "ext": ext})
        print(f"  pod{n} {pid} ssh={ip}:{sshp} ext={ext}")
    return pods


def ssh(pod, cmd, bg=False, timeout=120):
    full = ["ssh", "-i", KEY, *SSHOPT, "-p", str(pod["ssh"]), f"root@{pod['ip']}", cmd]
    if bg:
        return subprocess.Popen(full, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return subprocess.run(full, capture_output=True, text=True, timeout=timeout)


def sync(pod):
    ssh(pod, "rm -rf /workspace/ce && mkdir -p /workspace/ce")
    tar = subprocess.Popen(["tar","czf","-","-C","/home/watchtower/circuit-dllm","engine"], stdout=subprocess.PIPE)
    subprocess.run(["ssh","-i",KEY,*SSHOPT,"-p",str(pod["ssh"]),f"root@{pod['ip']}",
                    "tar xzf - -C /workspace/ce"], stdin=tar.stdout, check=True)
    tar.wait()


def ext_of(pod, port):
    ip, p = pod["ext"][port]
    return ip, p


def _tmux(pod, sess, inner):
    # tmux fully detaches the process so it survives the launching SSH session closing
    # (setsid did not — the control plane died when the harness SSH dropped).
    ssh(pod, f"tmux kill-session -t {sess} 2>/dev/null; cd /workspace/ce && "
             f"tmux new-session -d -s {sess} '{inner}'")


def launch_control(pod):
    cip, cpt = ext_of(pod, CONTROL_PORT)
    env = (f"CIRCUIT_ROLE=control CIRCUIT_MESH=1 CIRCUIT_MESH_LAYERS={LAYERS} CIRCUIT_MESH_STAGES=1 "
           f"CIRCUIT_MESH_REPLICATION=2 CIRCUIT_MESH_VERIFY_SIG=1 CIRCUIT_MESH_DEAD_AFTER=30 "
           f"CIRCUIT_KEY={CKEY} CIRCUIT_CONTROL_HOST=0.0.0.0 CIRCUIT_CONTROL_PORT={CONTROL_PORT} "
           f"CIRCUIT_COORDINATOR_ADVERTISE={cip} CIRCUIT_REAP_INTERVAL=5")
    _tmux(pod, "ctl", f"{env} python3 -u -m engine.api > /tmp/control.log 2>&1")
    return f"http://{cip}:{cpt}"


def launch_holder(pod, control_url):
    hip, hpt = ext_of(pod, HOLDER_PORT)
    inner = (f"CIRCUIT_KEY={CKEY} python3 -u -m engine.stage_worker --port {HOLDER_PORT} --model {MODEL} "
             f"--device cuda --host 0.0.0.0 --control-url {control_url} --advertise-host {hip} "
             f"--advertise-port {hpt} --capacity-layers {LAYERS} --hb-interval 5 "
             f"--node-key-file /workspace/h.hex > /tmp/holder.log 2>&1")
    _tmux(pod, "hold", inner)


def launch_orch(pod, control_url):
    oip, opt = ext_of(pod, ORCH_PORT)
    env = (f"CIRCUIT_ROLE=orchestrator CIRCUIT_CONTROL_URL={control_url} CIRCUIT_MODEL={MODEL} "
           f"CIRCUIT_DRAFT={DRAFT} CIRCUIT_KEY={CKEY} CIRCUIT_DEVICE=cuda "
           f"CIRCUIT_API_HOST=0.0.0.0 CIRCUIT_API_PORT={ORCH_PORT} CIRCUIT_ORCH_ADVERTISE={oip} "
           f"CIRCUIT_HEARTBEAT_INTERVAL=5 CIRCUIT_NODE_KEY_FILE=/workspace/o.hex")
    _tmux(pod, "orch", f"{env} python3 -u -m engine.api > /tmp/orch.log 2>&1")
    return f"http://{oip}:{opt}"


def http_json(url, obj=None, timeout=30):
    data = json.dumps(obj).encode() if obj is not None else None
    req = urllib.request.Request(url, data=data,
        headers={"Content-Type": "application/json"}, method=("POST" if obj is not None else "GET"))
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def wait(label, fn, tries=60, gap=4):
    for i in range(tries):
        try:
            if fn():
                print(f"  {label}: ready (~{i*gap}s)"); return True
        except Exception:
            pass
        time.sleep(gap)
    print(f"  {label}: TIMEOUT"); return False


def stream_tokens(orch_url, max_tokens):
    """Fire one stream:true completion, return (tokens, seconds)."""
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": PROMPT}],
                       "max_tokens": max_tokens, "stream": True}).encode()
    req = urllib.request.Request(orch_url + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time(); toks = 0
    with urllib.request.urlopen(req, timeout=300) as r:
        for raw in r:
            s = raw.decode("utf-8", "replace").strip()
            if s.startswith("data:") and '"delta"' in s and '"content"' in s:
                try:
                    j = json.loads(s[5:].strip())
                    if j["choices"][0]["delta"].get("content"):
                        toks += 1
                except Exception:
                    pass
    return toks, time.time() - t0


def sweep_config(orch_urls, concurrency, max_tokens=64, label=""):
    """Fire `concurrency` streaming requests round-robin across orch_urls; aggregate tok/s."""
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(stream_tokens, orch_urls[i % len(orch_urls)], max_tokens)
                for i in range(concurrency)]
        res = [f.result() for f in futs]
    wall = time.time() - t0
    total = sum(t for t, _ in res)
    return total / wall if wall else 0, total, wall


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "validate"
    pods = load_pods()
    print("syncing engine to all pods...")
    for p in pods:
        sync(p)

    control_url = launch_control(pods[0])
    print(f"control plane: {control_url}")
    wait("control plane", lambda: http_json(control_url + "/health", timeout=6).get("ok"), tries=30, gap=3)

    if mode == "validate":
        launch_holder(pods[0], LOCAL_CONTROL)        # co-located with the control plane → localhost
        wait("coverage", lambda: http_json(control_url + "/health", timeout=6)["coverage_ok"], tries=80, gap=5)
        orch_url = launch_orch(pods[2], control_url)
        wait("orchestrator", lambda: http_json(orch_url + "/health", timeout=6).get("role") == "orchestrator", tries=80, gap=4)
        print(f"\ncross-pod completion via {orch_url} (orchestrator on pod2 -> holder on pod0):")
        out = http_json(orch_url + "/v1/chat/completions",
                        {"model": MODEL, "messages": [{"role": "user", "content": "The capital of France is"}],
                         "max_tokens": 16, "stream": False})
        print("  ->", out["choices"][0]["message"]["content"])
        print("VALIDATE OK — scattered cross-pod relay works" if out["choices"][0]["message"]["content"] else "FAIL")
        return

    # sweep: 2 holders + 2 orchestrators
    launch_holder(pods[0], LOCAL_CONTROL)            # co-located with the control plane → localhost
    launch_holder(pods[1], control_url)              # remote → external proxy address
    wait("coverage (2 holders)", lambda: http_json(control_url + "/health", timeout=6)["coverage_ok"], tries=90, gap=5)
    o2 = launch_orch(pods[2], control_url)
    o3 = launch_orch(pods[3], control_url)
    wait("orch pod2", lambda: http_json(o2 + "/health", timeout=6).get("role") == "orchestrator", tries=80, gap=4)
    wait("orch pod3", lambda: http_json(o3 + "/health", timeout=6).get("role") == "orchestrator", tries=80, gap=4)

    print("\n=== THROUGHPUT SWEEP (1.5B target / 0.5B draft, replication=2, scattered) ===")
    print(f"{'conc':>5} | {'1-orch tok/s':>13} | {'2-orch tok/s':>13} | {'speedup':>8}")
    print("-" * 50)
    for c in [1, 2, 4, 8]:
        r1, _, _ = sweep_config([o2], c)              # all load on ONE orchestrator (the funnel)
        time.sleep(2)
        r2, _, _ = sweep_config([o2, o3], c)          # split across BOTH orchestrators
        print(f"{c:>5} | {r1:>13.1f} | {r2:>13.1f} | {r2/r1 if r1 else 0:>7.2f}x")
    print("\n2-orch > 1-orch at high concurrency => the orchestrator funnel is real and removed.")


if __name__ == "__main__":
    main()
