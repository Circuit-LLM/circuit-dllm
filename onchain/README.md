# circuit mesh_registry — on-chain control-plane anchor

The Solana program that anchors the Circuit DLLM control plane. It holds only the
**slow, authoritative** state from [`../docs/ONCHAIN_CONTROL_PLANE.md`](../docs/ONCHAIN_CONTROL_PLANE.md):

- **`MeshConfig`** (one PDA, seeds `["mesh_config"]`) — the topology contract: `model_fp`,
  `num_layers`, `replication`, the slot layout (a validated partition of `[0, num_layers)`),
  `version`, plus the `authority` and `auditor` keys.
- **`Node`** (one PDA per identity, seeds `["node", node_pubkey]`) — membership: `role`,
  `trust`, `banned`, `payout_wallet`, `stake_pool`, timestamps. A node can only ever write
  **its own** record (the PDA is seeded by its pubkey).

Fast state (liveness, slot assignment, routes) is **never** here — it's derived in-mesh
(heartbeats + rendezvous hashing). This program is touched zero times per request.

## Authority model

| Action | Who |
|---|---|
| `initialize_config` / `update_config` / `set_auditor` / `set_authority` | `authority` (operator/governance) |
| `register_node` / `update_node` / `deregister_node` | the node itself (PDA seed) |
| `set_trust` / `set_ban` | `auditor` (the probation→trusted / evict flow, on-chain) |

`initialize_config` makes the signer both `authority` and `auditor`; split them with `set_auditor`.

## Toolchain

- Solana CLI (Agave) `4.0.2`, `cargo-build-sbf` / platform-tools `v1.53`
- Anchor `anchor-cli 1.1.1` (via `avm`), `anchor-lang = "1.1.1"`, TS client `@coral-xyz/anchor ^0.32.1`

## Keypairs (NOT committed — see `.gitignore`)

`keys/` holds the program-id keypair and a local deployer. Regenerate if absent:

```bash
solana-keygen new --no-bip39-passphrase -o keys/mesh_registry-keypair.json   # program id
solana-keygen new --no-bip39-passphrase -o keys/deployer.json                # local/devnet wallet
# then sync declare_id! + Anchor.toml to the new program pubkey:
solana-keygen pubkey keys/mesh_registry-keypair.json
```

Program id (current): `BC2sxffu498cB8gUp3P5V5HuBLDsx9XCtJdEmnnGUvfe` (`keys/PROGRAM_ID.txt`).

## Build / test / deploy

```bash
# build (place the program keypair where anchor expects it first)
mkdir -p target/deploy && cp keys/mesh_registry-keypair.json target/deploy/
anchor build

# unit/integration tests on a throwaway local validator
yarn install          # or npm install
anchor test           # spins up solana-test-validator, deploys, runs tests/mesh_registry.ts

# devnet (Phase 1 ships to devnet only; mainnet is a later phase + an audit)
solana airdrop 2 keys/deployer.json --url devnet
anchor deploy --provider.cluster devnet --provider.wallet keys/deployer.json
```

## Readers

Later phases read these accounts with **pure RPC** (`getProgramAccounts` + `memcmp` filters +
8-byte Anchor discriminator), exactly like `circuit-node-client/lib/stakepoint.js` reads StakePoint
today — no heavy Anchor client on the read path.
