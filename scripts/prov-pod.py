#!/usr/bin/env python3
# Provision a clean L4 pod (250GB container disk, NO network volume) and hunt past
# oversubscribed machines (needs >MIN_FREE MiB free VRAM). Usage:
#   prov72.py <name> <role> <ports> <min_free_mib>
import json, subprocess, sys, time, urllib.request, urllib.error

def secret(n):
    return subprocess.check_output(["/home/watchtower/.openclaw/credentials/infisical-get.sh", n]).decode().strip()

RPK = secret("RUNPOD_API_KEY")
UA = {"User-Agent": "curl/8.5.0"}
AUTHID = "cmqoxrw9s009s4qofbvitg0qh"           # Circuit-LLM private GHCR auth
IMAGE  = "ghcr.io/circuit-llm/circuit-dllm:v2-indep"   # has bitsandbytes baked
NAME, ROLE, PORTS, MIN_FREE = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
CLOUD = sys.argv[5] if len(sys.argv) > 5 else "SECURE"      # SECURE | COMMUNITY
DISK  = int(sys.argv[6]) if len(sys.argv) > 6 else 250
GPUS  = sys.argv[7].split(",") if len(sys.argv) > 7 else ["NVIDIA L4"]
DC    = sys.argv[8] if len(sys.argv) > 8 and sys.argv[8] != "-" else None  # pin datacenter

def rest(method, path):
    req = urllib.request.Request("https://rest.runpod.io/v1/"+path,
            headers={"Authorization": "Bearer "+RPK, **UA}, method=method)
    try:
        return json.load(urllib.request.urlopen(req, timeout=40))
    except urllib.error.HTTPError as e:
        return {"http_error": e.code, "body": e.read().decode()[:300]}
    except Exception as e:
        return {"err": str(e)}

def gql(q, v=None):
    body = json.dumps({"query": q, "variables": v or {}}).encode()
    req = urllib.request.Request("https://api.runpod.io/graphql?api_key="+RPK,
            data=body, headers={"Content-Type": "application/json", **UA})
    try:
        return json.load(urllib.request.urlopen(req, timeout=60))
    except urllib.error.HTTPError as e:
        return {"http_error": e.code, "body": e.read().decode()[:400]}

pubkey = rest("GET", "pods/0e78t6cfl72z9y").get("env", {}).get("PUBLIC_KEY", "")
assert pubkey, "no PUBLIC_KEY"

_dcline = "dataCenterId:$dc," if DC else ""
MUT = ('''mutation($img:String!,$auth:String!,$pk:String!,$name:String!,$role:String!,$ports:String!,$cloud:CloudTypeEnum!,$gpu:String!,$disk:Int!,$dc:String){
  podFindAndDeployOnDemand(input:{
    cloudType:$cloud, gpuTypeId:$gpu, gpuCount:1, name:$name, ''' + _dcline + '''
    imageName:$img, containerRegistryAuthId:$auth, containerDiskInGb:$disk,
    volumeInGb:0, ports:$ports,
    env:[{key:"PUBLIC_KEY",value:$pk},{key:"CIRCUIT_ROLE",value:$role}]
  }){ id } }''')

def create():
    # try each GPU type until one has supply on this cloud
    for gpu in GPUS:
        r = gql(MUT, {"img":IMAGE,"auth":AUTHID,"pk":pubkey,"name":NAME,"role":ROLE,
                      "ports":PORTS,"cloud":CLOUD,"gpu":gpu,"disk":DISK,"dc":DC})
        try:
            return r["data"]["podFindAndDeployOnDemand"]["id"]
        except Exception:
            msg = json.dumps(r)[:160]
            if "SUPPLY_CONSTRAINT" not in msg:
                print(f"create failed ({gpu}):", msg)
    return None

def ssh_free(ip, port):
    try:
        out = subprocess.check_output(
            ["ssh","-o","ConnectTimeout=12","-o","StrictHostKeyChecking=no",
             "-o","UserKnownHostsFile=/dev/null","-i","/home/watchtower/.ssh/id_ed25519",
             "-p",str(port),"root@"+ip,
             "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=20).decode().strip()
        return int(out.split("\n")[0])
    except Exception:
        return -1

for attempt in range(1, 9):
    pid = create()
    if not pid:
        print(f"att {attempt}: create failed"); time.sleep(5); continue
    # poll for RUNNING + ssh port
    ip = sshport = None
    for _ in range(40):
        time.sleep(6)
        d = rest("GET", "pods/"+pid)
        if d.get("desiredStatus") == "RUNNING" and d.get("publicIp"):
            pm = d.get("portMappings") or {}
            if pm.get("22"):
                ip, sshport = d["publicIp"], pm["22"]; break
    if not ip:
        print(f"att {attempt}: {pid} never got ssh; destroying"); rest("DELETE","pods/"+pid); continue
    # wait for sshd, then check free VRAM
    free = -1
    for _ in range(20):
        time.sleep(8)
        free = ssh_free(ip, sshport)
        if free >= 0: break
    print(f"att {attempt}: {pid} ip={ip} ssh={sshport} free={free}MiB")
    if free >= MIN_FREE:
        pm = rest("GET","pods/"+pid).get("portMappings") or {}
        print("CLEAN_POD", json.dumps({"id":pid,"ip":ip,"ssh":sshport,"ports":pm}))
        sys.exit(0)
    print(f"  oversubscribed (<{MIN_FREE}); destroying + retrying")
    rest("DELETE","pods/"+pid)

print("FAILED: no clean L4 after retries")
sys.exit(1)
