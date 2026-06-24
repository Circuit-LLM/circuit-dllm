# Set up a RunPod GPU node

Run a Circuit mesh node on a [RunPod](https://runpod.io) cloud GPU. RunPod gives every pod a public
IP, so there's no NAT/relay to deal with — the coordinator can reach your node directly. For a home
desktop instead, see [SETUP_DESKTOP.md](SETUP_DESKTOP.md).

The clean way on RunPod is to deploy a pod **from the node image** (`ghcr.io/circuit-llm/gpu-node`).
RunPod pulls the image and runs its entrypoint on the GPU — no Docker-in-Docker, no installer.

---

## 0. What you need

- A RunPod account with credits.
- A **Solana wallet** for CIRC earnings (optional).
- The image is **public**, so no registry credentials are required.

---

## 1. Deploy a pod from the image (Console)

1. **Pods → Deploy** → pick a GPU (an **L4 / RTX A4000 / RTX 4000 Ada** is plenty; more VRAM = more
   layers = more earnings).
2. **Container image:** choose **"Custom"** and set it to:
   ```
   ghcr.io/circuit-llm/gpu-node:latest
   ```
3. **Container disk:** **40 GB** or more (the image + your model slice).
4. **Expose a TCP port:** add **`19210`** under *TCP Port Mappings*. The coordinator dials this port,
   and the entrypoint auto-advertises RunPod's public IP + the mapped port (`RUNPOD_PUBLIC_IP` /
   `RUNPOD_TCP_PORT_19210`), so it just works.
5. **Environment variables:**
   | Key | Value |
   |-----|-------|
   | `CIRCUIT_CONTROL_URL` | `https://node.circuitllm.xyz` |
   | `CIRCUIT_PAYOUT_WALLET` | `<your Solana address>` (optional) |
   | `CIRCUIT_REGION` | e.g. `na-east` (optional, helps routing) |
6. **Deploy.** The pod pulls the image (~minutes the first time), then the entrypoint detects the
   GPU, sizes its capacity from VRAM, registers, downloads its assigned slice, and serves.

> Don't add a network volume; the container disk is fine for a node. Leave the rest default.

---

## 2. Confirm it joined

- **Pod logs** (Console → your pod → *Logs*): look for `GPU: … → capacity N layers`, then
  `joined mesh … layers a:b`, then `ready (serving)`.
- **Live topology** — your node should appear as a holder:
  ```bash
  curl -s https://node.circuitllm.xyz/topology | python3 -m json.tool
  ```
  A fresh node shows `state: joining` (downloading its slice) then `state: ready`, and `trust:
  probation` until the verifier promotes it.

---

## 3. Deploy via API / CLI (optional, for automation)

RunPod's GraphQL API deploys the same image headlessly — useful for scripting a fleet. Minimal
mutation (see `scripts/prov-pod.py` for a complete, retrying example):

```graphql
mutation {
  podFindAndDeployOnDemand(input:{
    cloudType: COMMUNITY, gpuTypeId: "NVIDIA L4", gpuCount: 1,
    name: "circuit-node",
    imageName: "ghcr.io/circuit-llm/gpu-node:latest",
    containerDiskInGb: 40, volumeInGb: 0,
    ports: "19210/tcp",
    env: [
      {key:"CIRCUIT_CONTROL_URL", value:"https://node.circuitllm.xyz"},
      {key:"CIRCUIT_PAYOUT_WALLET", value:"<your-wallet>"}
    ]
  }){ id }
}
```
POST it to `https://api.runpod.io/graphql?api_key=<RUNPOD_API_KEY>`. The image is public, so no
`containerRegistryAuthId` is needed.

---

## 4. Costs, earnings, lifecycle

- **You pay RunPod** for the GPU by the hour and **earn CIRC** for the inference work your node
  serves (`layers × tokens`). Pick a GPU whose hourly cost your expected earnings justify — bigger,
  pricier cards hold more layers but cost more; commodity cards (L4-class) are the sweet spot the
  network is built around.
- **Trust:** a new node serves on **probation** (never primary for a token) and is promoted to
  **trusted** after it passes correctness challenges ([VERIFICATION.md](VERIFICATION.md)). Just keep
  it running.
- **Stop** the pod from the Console when you're done (you're billed while it runs). A stopped/restarted
  pod re-registers; for a stable identity across restarts, mount a small volume at `/var/lib/circuit`.
- **Update:** redeploy with the same image tag to pull the latest, or pin the digest for stability.

---

## Troubleshooting

| Symptom | Fix |
|--------|-----|
| Pod `RUNNING` but never in `/topology` | Check pod logs for an image-pull or GPU error; ensure `CIRCUIT_CONTROL_URL` is set. |
| Registers but stuck `joining` | It's downloading its model slice — wait a few minutes. |
| Coordinator can't reach the node | You didn't expose **TCP 19210** — add it under Port Mappings and redeploy. |
| `model_fp mismatch` | Old image tag — redeploy `:latest`. |
