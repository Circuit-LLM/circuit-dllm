// Anchor migration hook. Phase 1 deploys the program only; the MeshConfig is
// initialized + the registry mirrored in Phase 2 (see docs/ONCHAIN_CONTROL_PLANE.md §8),
// deliberately not here, so a redeploy never clobbers live config.
const anchor = require("@coral-xyz/anchor");

module.exports = async function (provider: anchor.AnchorProvider) {
  anchor.setProvider(provider);
  // intentionally a no-op beyond deploy — config init is a separate, gated step.
};
