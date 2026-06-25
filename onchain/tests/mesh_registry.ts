import * as anchor from "@coral-xyz/anchor";
import { Program } from "@coral-xyz/anchor";
import { MeshRegistry } from "../target/types/mesh_registry";
import { PublicKey, Keypair, LAMPORTS_PER_SOL, SystemProgram } from "@solana/web3.js";
import { assert } from "chai";

// Mirrors the spec's prod topology: Qwen2.5-72B-AWQ, 80 layers, two slots.
const MODEL_FP = "qwen2.5-72b-awq";
const NUM_LAYERS = 80;
const REPLICATION = 2;
const SLOTS = [
  { start: 0, end: 40 },
  { start: 40, end: 80 },
];

describe("mesh_registry", () => {
  const provider = anchor.AnchorProvider.env();
  anchor.setProvider(provider);
  const program = anchor.workspace.meshRegistry as Program<MeshRegistry>;
  const authority = provider.wallet as anchor.Wallet; // == config authority/auditor at init

  const configPda = PublicKey.findProgramAddressSync(
    [Buffer.from("mesh_config")],
    program.programId
  )[0];

  const nodePda = (owner: PublicKey) =>
    PublicKey.findProgramAddressSync(
      [Buffer.from("node"), owner.toBuffer()],
      program.programId
    )[0];

  // Fund a fresh keypair so it can pay rent for its own Node PDA.
  async function fund(kp: Keypair, sol = 1) {
    const sig = await provider.connection.requestAirdrop(kp.publicKey, sol * LAMPORTS_PER_SOL);
    const bh = await provider.connection.getLatestBlockhash();
    await provider.connection.confirmTransaction({ signature: sig, ...bh });
  }

  const holder = Keypair.generate();   // a holder node identity
  const orch = Keypair.generate();     // an orchestrator node identity
  const auditor = Keypair.generate();  // promoted to auditor mid-suite
  const stranger = Keypair.generate(); // unauthorized key

  before(async () => {
    await Promise.all([fund(holder), fund(orch), fund(auditor), fund(stranger)]);
  });

  // ---- MeshConfig ---------------------------------------------------------

  it("initializes the singleton config (authority == auditor at init)", async () => {
    await program.methods
      .initializeConfig(MODEL_FP, NUM_LAYERS, REPLICATION, SLOTS)
      .accounts({ config: configPda, authority: authority.publicKey, systemProgram: SystemProgram.programId })
      .rpc();

    const cfg = await program.account.meshConfig.fetch(configPda);
    assert.equal(cfg.modelFp, MODEL_FP);
    assert.equal(cfg.numLayers, NUM_LAYERS);
    assert.equal(cfg.replication, REPLICATION);
    assert.equal(cfg.version, 1);
    assert.equal(cfg.slots.length, 2);
    assert.deepEqual([cfg.slots[0].start, cfg.slots[0].end], [0, 40]);
    assert.ok(cfg.authority.equals(authority.publicKey));
    assert.ok(cfg.auditor.equals(authority.publicKey));
  });

  it("rejects a re-init of the singleton", async () => {
    try {
      await program.methods
        .initializeConfig(MODEL_FP, NUM_LAYERS, REPLICATION, SLOTS)
        .accounts({ config: configPda, authority: authority.publicKey, systemProgram: SystemProgram.programId })
        .rpc();
      assert.fail("re-init should fail");
    } catch (e) {
      // account already in use
      assert.ok(String(e).length > 0);
    }
  });

  it("authority can update the topology and bumps version", async () => {
    const newSlots = [
      { start: 0, end: 20 },
      { start: 20, end: 40 },
      { start: 40, end: 60 },
      { start: 60, end: 80 },
    ];
    await program.methods
      .updateConfig(MODEL_FP, NUM_LAYERS, 3, newSlots)
      .accounts({ config: configPda, authority: authority.publicKey })
      .rpc();
    const cfg = await program.account.meshConfig.fetch(configPda);
    assert.equal(cfg.version, 2);
    assert.equal(cfg.replication, 3);
    assert.equal(cfg.slots.length, 4);
  });

  it("rejects a layout with a gap (bad coverage)", async () => {
    const gapped = [{ start: 0, end: 30 }, { start: 40, end: 80 }]; // 30..40 uncovered
    try {
      await program.methods
        .updateConfig(MODEL_FP, NUM_LAYERS, REPLICATION, gapped)
        .accounts({ config: configPda, authority: authority.publicKey })
        .rpc();
      assert.fail("gapped layout should be rejected");
    } catch (e) {
      assert.match(String(e), /BadCoverage|partition/);
    }
  });

  it("rejects a layout with an overlap (bad coverage)", async () => {
    const overlap = [{ start: 0, end: 50 }, { start: 40, end: 80 }];
    try {
      await program.methods
        .updateConfig(MODEL_FP, NUM_LAYERS, REPLICATION, overlap)
        .accounts({ config: configPda, authority: authority.publicKey })
        .rpc();
      assert.fail("overlapping layout should be rejected");
    } catch (e) {
      assert.match(String(e), /BadCoverage|partition/);
    }
  });

  it("rejects a non-authority editing the config", async () => {
    try {
      await program.methods
        .updateConfig(MODEL_FP, NUM_LAYERS, REPLICATION, SLOTS)
        .accounts({ config: configPda, authority: stranger.publicKey })
        .signers([stranger])
        .rpc();
      assert.fail("stranger must not edit config");
    } catch (e) {
      assert.match(String(e), /NotAuthority|has_one|ConstraintHasOne|unknown signer|Signature/);
    }
  });

  it("authority delegates the auditor role to a separate key", async () => {
    await program.methods
      .setAuditor(auditor.publicKey)
      .accounts({ config: configPda, authority: authority.publicKey })
      .rpc();
    const cfg = await program.account.meshConfig.fetch(configPda);
    assert.ok(cfg.auditor.equals(auditor.publicKey));
    assert.ok(cfg.authority.equals(authority.publicKey)); // authority unchanged
  });

  // ---- Node self-registration --------------------------------------------

  it("a node self-registers its own PDA (probation, not banned)", async () => {
    await program.methods
      .registerNode({ holder: {} }, holder.publicKey, PublicKey.default)
      .accounts({ node: nodePda(holder.publicKey), signer: holder.publicKey, systemProgram: SystemProgram.programId })
      .signers([holder])
      .rpc();

    const n = await program.account.node.fetch(nodePda(holder.publicKey));
    assert.ok(n.node.equals(holder.publicKey));
    assert.deepEqual(n.role, { holder: {} });
    assert.deepEqual(n.trust, { probation: {} });
    assert.equal(n.banned, false);
    assert.ok(n.payoutWallet.equals(holder.publicKey));
    assert.ok(n.joinedAt.toNumber() > 0);
  });

  it("an orchestrator self-registers independently", async () => {
    await program.methods
      .registerNode({ orchestrator: {} }, orch.publicKey, PublicKey.default)
      .accounts({ node: nodePda(orch.publicKey), signer: orch.publicKey, systemProgram: SystemProgram.programId })
      .signers([orch])
      .rpc();
    const n = await program.account.node.fetch(nodePda(orch.publicKey));
    assert.deepEqual(n.role, { orchestrator: {} });
  });

  it("a node updates its own declared role / payout (self-only)", async () => {
    const newPayout = Keypair.generate().publicKey;
    await program.methods
      .updateNode({ holder: {} }, newPayout, PublicKey.default)
      .accounts({ node: nodePda(holder.publicKey), signer: holder.publicKey })
      .signers([holder])
      .rpc();
    const n = await program.account.node.fetch(nodePda(holder.publicKey));
    assert.ok(n.payoutWallet.equals(newPayout));
  });

  // ---- Auditor trust/ban (the core Phase-1 DoD) --------------------------

  it("the auditor promotes a node probation -> trusted", async () => {
    await program.methods
      .setTrust({ trusted: {} })
      .accounts({ config: configPda, node: nodePda(holder.publicKey), auditor: auditor.publicKey })
      .signers([auditor])
      .rpc();
    const n = await program.account.node.fetch(nodePda(holder.publicKey));
    assert.deepEqual(n.trust, { trusted: {} });
  });

  it("a non-auditor cannot change trust", async () => {
    try {
      await program.methods
        .setTrust({ probation: {} })
        .accounts({ config: configPda, node: nodePda(holder.publicKey), auditor: stranger.publicKey })
        .signers([stranger])
        .rpc();
      assert.fail("stranger must not flip trust");
    } catch (e) {
      assert.match(String(e), /NotAuditor/);
    }
  });

  it("the node itself cannot self-promote trust", async () => {
    try {
      await program.methods
        .setTrust({ trusted: {} })
        .accounts({ config: configPda, node: nodePda(holder.publicKey), auditor: holder.publicKey })
        .signers([holder])
        .rpc();
      assert.fail("a node must not be its own auditor");
    } catch (e) {
      assert.match(String(e), /NotAuditor/);
    }
  });

  it("the auditor bans and un-bans a node", async () => {
    await program.methods
      .setBan(true)
      .accounts({ config: configPda, node: nodePda(orch.publicKey), auditor: auditor.publicKey })
      .signers([auditor])
      .rpc();
    let n = await program.account.node.fetch(nodePda(orch.publicKey));
    assert.equal(n.banned, true);

    await program.methods
      .setBan(false)
      .accounts({ config: configPda, node: nodePda(orch.publicKey), auditor: auditor.publicKey })
      .signers([auditor])
      .rpc();
    n = await program.account.node.fetch(nodePda(orch.publicKey));
    assert.equal(n.banned, false);
  });

  // ---- Graceful leave -----------------------------------------------------

  it("a node closes its own record (rent reclaimed)", async () => {
    await program.methods
      .deregisterNode()
      .accounts({ node: nodePda(orch.publicKey), signer: orch.publicKey })
      .signers([orch])
      .rpc();
    const info = await provider.connection.getAccountInfo(nodePda(orch.publicKey));
    assert.isNull(info);
  });
});
