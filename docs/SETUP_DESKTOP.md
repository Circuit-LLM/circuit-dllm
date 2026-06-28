# Set up a desktop / home GPU node

Turn a desktop with an NVIDIA GPU into a Circuit mesh node — it holds a slice of the decentralized
Qwen2.5-72B, serves real traffic, and earns CIRC. Works on **Linux** and **Windows (via WSL2)**.
For a cloud GPU instead, see [SETUP_RUNPOD.md](SETUP_RUNPOD.md).

> You never run a `docker` command yourself — the installer handles Docker, the NVIDIA toolkit, and
> the container. Plan ~20–40 GB of free disk (the image + your assigned model slice).

---

## 0. What you need

- An **NVIDIA GPU**. 8 GB VRAM is the practical minimum; more VRAM → more layers → more earnings.
- A reasonably fast, always-on internet connection (your node serves live requests).
- A **Solana wallet address** to receive CIRC (optional — you can run unpaid first).

---

## 1. Linux

### 1.1 Confirm the GPU driver
```bash
nvidia-smi
```
You should see your GPU and driver version. If `nvidia-smi` is missing:
- **Ubuntu/Debian:** `sudo ubuntu-drivers autoinstall` then reboot.
- Then re-run `nvidia-smi` to confirm.

### 1.2 Run the installer

**Linux / WSL2** (inside a bash shell):
```bash
curl -fsSL https://circuitllm.xyz/join | bash
```

**Windows** — run in an **Administrator PowerShell** (it installs WSL2 + Ubuntu if needed, then runs
the installer inside it; the GPU passes through from the NVIDIA *Windows* driver, no driver inside WSL):
```powershell
irm https://circuitllm.xyz/join.ps1 | iex
```
> Don't paste the `curl … | bash` line into PowerShell — there `curl` is an alias for
> `Invoke-WebRequest` and just prints the HTTP response (the `$'\r'` / `StatusCode:` errors). Use the
> `irm … | iex` line, or run the bash line from inside a WSL/Ubuntu shell.
It will:
1. detect your GPU,
2. install **Docker** + the **NVIDIA Container Toolkit** if they're missing,
3. ask for your **payout wallet** (press Enter to skip and run unpaid),
4. pull the GPU image and start it as an **auto-restarting** container.

First start downloads your assigned model slice (a few GB) — give it a few minutes.

### 1.3 Confirm it joined
```bash
docker logs -f circuit-gpu-node
```
Look for `GPU: … → capacity N layers`, then `joined mesh … layers a:b`. You can also confirm your
node shows up in the live topology:
```bash
curl -s https://node.circuitllm.xyz/topology | python3 -m json.tool
```

---

## 2. Windows (WSL2)

Windows joins through **WSL2** — the GPU passes through, no dual-boot, no driver inside WSL.

1. **Install WSL2** (PowerShell as Administrator), then reboot:
   ```powershell
   wsl --install
   ```
2. **Install the NVIDIA *Windows* driver** — the normal Game-Ready/Studio driver. It provides CUDA
   to WSL. **Do not** install a GPU driver *inside* WSL.
3. Open the **Ubuntu (WSL)** terminal from the Start menu and run the same one-liner:
   ```bash
   curl -fsSL https://circuitllm.xyz/join | bash
   ```
4. Confirm with `nvidia-smi` (should list your GPU inside WSL) and `docker logs -f circuit-gpu-node`.

> Prefer one click? In an **Administrator PowerShell**, run `irm https://circuitllm.xyz/join.ps1 | iex`
> — it sets up WSL2 and runs the installer for you.

---

## 3. Home routers & NAT (important)

The coordinator must be able to **reach your node** to send it work. Two cases:

- **You can port-forward** — forward **TCP 19210** on your router to this machine, then re-run the
  installer with `CIRCUIT_ADVERTISE_HOST=<your public IP>` set. Simplest if your router allows it.
- **You can't / won't port-forward** (most home setups) — use the **relay**: the node dials *out* to
  a relay and the coordinator reaches it through that connection, no router config:
  ```bash
  CIRCUIT_RELAY_URL=relay.circuitllm.xyz:18940 curl -fsSL https://circuitllm.xyz/join | bash
  ```
  The relay only ever sees encrypted bytes — it can't read your traffic (see [RELAY.md](RELAY.md)).

If you set neither, the installer advertises your public IP; that works only if your GPU is directly
reachable inbound (rare on home networks).

---

## 4. Earnings, updates, and stopping

- **Earnings.** You earn CIRC proportional to `layers × tokens` your node serves. Set a wallet at
  install time (or re-run the installer with `CIRCUIT_PAYOUT_WALLET=<addr>`).
- **Trust.** A new node starts on **probation** (never the primary for a token) and is promoted to
  **trusted** once it passes correctness checks against a trusted replica ([VERIFICATION.md](VERIFICATION.md)).
  Nothing to do — just keep it running and online.
- **Update:** `docker pull ghcr.io/circuit-llm/gpu-node:latest && bash <(curl -fsSL https://circuitllm.xyz/join)`
- **Stop / start:** `docker stop circuit-gpu-node` · `docker start circuit-gpu-node`
- **Logs:** `docker logs -f circuit-gpu-node`
- **Identity** persists in a Docker volume (`circuit-gpu`), so restarts keep the same node id +
  downloaded slice. Removing that volume re-registers you as a brand-new node.

---

## Troubleshooting

| Symptom | Fix |
|--------|-----|
| `no NVIDIA GPU visible inside the container` | Install the NVIDIA Container Toolkit (the installer does this); on WSL2, install the **Windows** driver, not one inside WSL. |
| Node registers then never reaches `ready` | It's downloading its slice — give it a few minutes; check `docker logs`. |
| Registers but the coordinator can't reach it | NAT — use `CIRCUIT_RELAY_URL` or port-forward 19210 (section 3). |
| `model_fp mismatch` rejection | Your image is stale; `docker pull` the latest and re-run. |
