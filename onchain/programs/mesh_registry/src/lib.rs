//! mesh_registry — the on-chain anchor for the Circuit DLLM control plane.
//!
//! Holds only the SLOW, authoritative, security-critical state (see
//! `docs/ONCHAIN_CONTROL_PLANE.md`): the topology contract (`MeshConfig`) and
//! per-identity membership (`Node`: role, trust, ban, payout, stake pool).
//!
//! Deliberately NOT here (too frequent → in-mesh per the spec): a node's
//! endpoint (host:port) and its liveness. Routing/liveness are derived locally
//! on every node via heartbeats + rendezvous hashing; this program is touched
//! zero times per request, only on join / trust-change / layout edit.
//!
//! Authority model:
//!   * `authority` — operator/governance: edits the topology, sets the auditor,
//!     can hand off authority. One per mesh.
//!   * `auditor`   — the only key that may flip a node's trust/ban (the
//!     probation→trusted / evict flow from docs/VERIFICATION.md, on-chain).
//!   * a `Node` PDA is seeded by the node's own pubkey, so a node can only
//!     create/update/close ITS OWN record (trustless self-registration).

use anchor_lang::prelude::*;

declare_id!("BC2sxffu498cB8gUp3P5V5HuBLDsx9XCtJdEmnnGUvfe");

/// Max bytes for the model fingerprint string (e.g. "qwen2.5-72b-awq").
const MAX_MODEL_FP: usize = 64;
/// Max slots in a layout (80 layers / smallest sane slice ≫ headroom).
const MAX_SLOTS: usize = 32;

#[program]
pub mod mesh_registry {
    use super::*;

    /// Create the singleton `MeshConfig`. The signer becomes both `authority`
    /// and the initial `auditor` (split later via `set_auditor`). `init` makes
    /// this callable exactly once for the canonical PDA.
    pub fn initialize_config(
        ctx: Context<InitializeConfig>,
        model_fp: String,
        num_layers: u16,
        replication: u8,
        slots: Vec<SlotRange>,
    ) -> Result<()> {
        validate_topology(&model_fp, num_layers, replication, &slots)?;
        let cfg = &mut ctx.accounts.config;
        cfg.authority = ctx.accounts.authority.key();
        cfg.auditor = ctx.accounts.authority.key();
        cfg.model_fp = model_fp;
        cfg.num_layers = num_layers;
        cfg.replication = replication;
        cfg.slots = slots;
        cfg.version = 1;
        cfg.bump = ctx.bumps.config;
        emit!(ConfigUpdated {
            version: cfg.version,
            num_layers: cfg.num_layers,
            replication: cfg.replication,
            slot_count: cfg.slots.len() as u8,
        });
        Ok(())
    }

    /// Overwrite the mutable topology fields and bump `version`. Authority-only.
    /// Readers detect a layout change by the version increment.
    pub fn update_config(
        ctx: Context<UpdateConfig>,
        model_fp: String,
        num_layers: u16,
        replication: u8,
        slots: Vec<SlotRange>,
    ) -> Result<()> {
        validate_topology(&model_fp, num_layers, replication, &slots)?;
        let cfg = &mut ctx.accounts.config;
        cfg.model_fp = model_fp;
        cfg.num_layers = num_layers;
        cfg.replication = replication;
        cfg.slots = slots;
        cfg.version = cfg.version.checked_add(1).ok_or(MeshError::VersionOverflow)?;
        emit!(ConfigUpdated {
            version: cfg.version,
            num_layers: cfg.num_layers,
            replication: cfg.replication,
            slot_count: cfg.slots.len() as u8,
        });
        Ok(())
    }

    /// Transfer config authority (governance handoff / key rotation).
    pub fn set_authority(ctx: Context<AdminConfig>, new_authority: Pubkey) -> Result<()> {
        ctx.accounts.config.authority = new_authority;
        Ok(())
    }

    /// Set the auditor allowed to flip node trust/ban. Authority-only.
    pub fn set_auditor(ctx: Context<AdminConfig>, new_auditor: Pubkey) -> Result<()> {
        ctx.accounts.config.auditor = new_auditor;
        Ok(())
    }

    /// A node creates ITS OWN membership record. The PDA is seeded by the
    /// signer's pubkey, so a node can only ever create its own. Starts on
    /// probation, not banned — the auditor promotes it once verified.
    pub fn register_node(
        ctx: Context<RegisterNode>,
        role: NodeRole,
        payout_wallet: Pubkey,
        stake_pool: Pubkey,
    ) -> Result<()> {
        let now = Clock::get()?.unix_timestamp;
        let node = &mut ctx.accounts.node;
        node.node = ctx.accounts.signer.key();
        node.role = role;
        node.trust = TrustLevel::Probation;
        node.banned = false;
        node.payout_wallet = payout_wallet;
        node.stake_pool = stake_pool;
        node.joined_at = now;
        node.updated_at = now;
        node.bump = ctx.bumps.node;
        emit!(NodeRegistered { node: node.node, role });
        Ok(())
    }

    /// A node updates its own declared role / payout wallet / stake pool.
    /// Self-only (enforced by the PDA seed). Does NOT touch trust/ban.
    pub fn update_node(
        ctx: Context<UpdateNode>,
        role: NodeRole,
        payout_wallet: Pubkey,
        stake_pool: Pubkey,
    ) -> Result<()> {
        let node = &mut ctx.accounts.node;
        node.role = role;
        node.payout_wallet = payout_wallet;
        node.stake_pool = stake_pool;
        node.updated_at = Clock::get()?.unix_timestamp;
        Ok(())
    }

    /// Auditor sets a node's trust level (probation ⇄ trusted). Auditor-only.
    pub fn set_trust(ctx: Context<AuditNode>, trust: TrustLevel) -> Result<()> {
        let node = &mut ctx.accounts.node;
        node.trust = trust;
        node.updated_at = Clock::get()?.unix_timestamp;
        emit!(NodeAudited { node: node.node, trust: node.trust, banned: node.banned });
        Ok(())
    }

    /// Auditor bans/unbans a node (eviction). Auditor-only. Routing drops a
    /// banned node regardless of liveness.
    pub fn set_ban(ctx: Context<AuditNode>, banned: bool) -> Result<()> {
        let node = &mut ctx.accounts.node;
        node.banned = banned;
        node.updated_at = Clock::get()?.unix_timestamp;
        emit!(NodeAudited { node: node.node, trust: node.trust, banned: node.banned });
        Ok(())
    }

    /// A node closes its own record (graceful leave); rent returns to it.
    /// Self-only (PDA seed). Auditors use `set_ban` to evict logically.
    pub fn deregister_node(_ctx: Context<DeregisterNode>) -> Result<()> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

/// A contiguous half-open layer range `[start, end)` assigned to one slot.
#[derive(AnchorSerialize, AnchorDeserialize, Clone, InitSpace)]
pub struct SlotRange {
    pub start: u16,
    pub end: u16,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, Copy, PartialEq, Eq, InitSpace)]
pub enum NodeRole {
    Orchestrator,
    Holder,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, Copy, PartialEq, Eq, InitSpace)]
pub enum TrustLevel {
    Probation,
    Trusted,
}

/// The singleton topology contract every node agrees on. One PDA per mesh.
#[account]
#[derive(InitSpace)]
pub struct MeshConfig {
    /// Operator/governance: edits topology, sets auditor, hands off authority.
    pub authority: Pubkey,
    /// The only key allowed to flip node trust/ban.
    pub auditor: Pubkey,
    /// Model fingerprint, e.g. "qwen2.5-72b-awq".
    #[max_len(MAX_MODEL_FP)]
    pub model_fp: String,
    /// Total transformer layers (e.g. 80).
    pub num_layers: u16,
    /// Target replicas per slot.
    pub replication: u8,
    /// The slot layout: a validated partition of `[0, num_layers)`.
    #[max_len(MAX_SLOTS)]
    pub slots: Vec<SlotRange>,
    /// Monotonic version, bumped on every edit (readers diff on this).
    pub version: u32,
    pub bump: u8,
}

/// Per-identity membership. One PDA per node, seeded by the node's pubkey.
#[account]
#[derive(InitSpace)]
pub struct Node {
    /// The node's ed25519 identity (== the key that created this record).
    pub node: Pubkey,
    pub role: NodeRole,
    pub trust: TrustLevel,
    pub banned: bool,
    /// Where CIRC payouts for this node are sent.
    pub payout_wallet: Pubkey,
    /// The StakePoint pool this node's stake is read from (stake itself is read
    /// off-chain via getProgramAccounts — never duplicated here).
    pub stake_pool: Pubkey,
    pub joined_at: i64,
    pub updated_at: i64,
    pub bump: u8,
}

// ---------------------------------------------------------------------------
// Contexts
// ---------------------------------------------------------------------------

#[derive(Accounts)]
pub struct InitializeConfig<'info> {
    #[account(
        init,
        payer = authority,
        space = 8 + MeshConfig::INIT_SPACE,
        seeds = [b"mesh_config"],
        bump
    )]
    pub config: Account<'info, MeshConfig>,
    #[account(mut)]
    pub authority: Signer<'info>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct UpdateConfig<'info> {
    #[account(mut, seeds = [b"mesh_config"], bump = config.bump, has_one = authority @ MeshError::NotAuthority)]
    pub config: Account<'info, MeshConfig>,
    pub authority: Signer<'info>,
}

/// Shared by `set_authority` / `set_auditor` — both authority-only.
#[derive(Accounts)]
pub struct AdminConfig<'info> {
    #[account(mut, seeds = [b"mesh_config"], bump = config.bump, has_one = authority @ MeshError::NotAuthority)]
    pub config: Account<'info, MeshConfig>,
    pub authority: Signer<'info>,
}

#[derive(Accounts)]
pub struct RegisterNode<'info> {
    #[account(
        init,
        payer = signer,
        space = 8 + Node::INIT_SPACE,
        seeds = [b"node", signer.key().as_ref()],
        bump
    )]
    pub node: Account<'info, Node>,
    #[account(mut)]
    pub signer: Signer<'info>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct UpdateNode<'info> {
    // Seed by the signer's key → a node can only update its own record.
    #[account(mut, seeds = [b"node", signer.key().as_ref()], bump = node.bump)]
    pub node: Account<'info, Node>,
    pub signer: Signer<'info>,
}

#[derive(Accounts)]
pub struct AuditNode<'info> {
    #[account(seeds = [b"mesh_config"], bump = config.bump)]
    pub config: Account<'info, MeshConfig>,
    // Seed-checked against its own stored identity so an arbitrary account
    // can't be passed as a Node.
    #[account(mut, seeds = [b"node", node.node.as_ref()], bump = node.bump)]
    pub node: Account<'info, Node>,
    #[account(constraint = auditor.key() == config.auditor @ MeshError::NotAuditor)]
    pub auditor: Signer<'info>,
}

#[derive(Accounts)]
pub struct DeregisterNode<'info> {
    #[account(
        mut,
        close = signer,
        seeds = [b"node", signer.key().as_ref()],
        bump = node.bump
    )]
    pub node: Account<'info, Node>,
    #[account(mut)]
    pub signer: Signer<'info>,
}

// ---------------------------------------------------------------------------
// Events  (orchestrators/gateway subscribe to these for push updates)
// ---------------------------------------------------------------------------

#[event]
pub struct ConfigUpdated {
    pub version: u32,
    pub num_layers: u16,
    pub replication: u8,
    pub slot_count: u8,
}

#[event]
pub struct NodeRegistered {
    pub node: Pubkey,
    pub role: NodeRole,
}

#[event]
pub struct NodeAudited {
    pub node: Pubkey,
    pub trust: TrustLevel,
    pub banned: bool,
}

// ---------------------------------------------------------------------------
// Validation + errors
// ---------------------------------------------------------------------------

/// The slot layout must be a clean partition of `[0, num_layers)` — no gaps,
/// no overlaps — so every node agrees on exactly which layers each slot owns.
fn validate_topology(
    model_fp: &str,
    num_layers: u16,
    replication: u8,
    slots: &[SlotRange],
) -> Result<()> {
    require!(model_fp.len() <= MAX_MODEL_FP, MeshError::ModelFpTooLong);
    require!(num_layers > 0, MeshError::BadNumLayers);
    require!(replication > 0, MeshError::BadReplication);
    require!(!slots.is_empty(), MeshError::NoSlots);
    require!(slots.len() <= MAX_SLOTS, MeshError::TooManySlots);
    for s in slots.iter() {
        require!(s.start < s.end && s.end <= num_layers, MeshError::BadSlotRange);
    }
    // Sort a copy by start, then require contiguity covering [0, num_layers).
    let mut sorted: Vec<&SlotRange> = slots.iter().collect();
    sorted.sort_by_key(|s| s.start);
    require!(sorted[0].start == 0, MeshError::BadCoverage);
    for i in 1..sorted.len() {
        require!(sorted[i].start == sorted[i - 1].end, MeshError::BadCoverage);
    }
    require!(sorted.last().unwrap().end == num_layers, MeshError::BadCoverage);
    Ok(())
}

#[error_code]
pub enum MeshError {
    #[msg("model fingerprint too long")]
    ModelFpTooLong,
    #[msg("num_layers must be > 0")]
    BadNumLayers,
    #[msg("replication must be > 0")]
    BadReplication,
    #[msg("no slots provided")]
    NoSlots,
    #[msg("too many slots")]
    TooManySlots,
    #[msg("slot range invalid: require start < end <= num_layers")]
    BadSlotRange,
    #[msg("slots must partition [0, num_layers) with no gaps or overlaps")]
    BadCoverage,
    #[msg("signer is not the config authority")]
    NotAuthority,
    #[msg("signer is not the configured auditor")]
    NotAuditor,
    #[msg("config version overflow")]
    VersionOverflow,
}
