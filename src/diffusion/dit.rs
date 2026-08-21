// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: Diffusion Transformer (DiT / Flux) backbone contract

use candle_core::{Result, Tensor};

/// Trait abstraction for modern Diffusion Transformer backbones (e.g. DiT, PixArt, Flux).
pub trait DiffusionTransformer {
    /// Forward pass through transformer blocks with text & time conditioning.
    fn forward(
        &self,
        latents: &Tensor,
        timestep: &Tensor,
        context: &Tensor,
        guidance: Option<&Tensor>,
    ) -> Result<Tensor>;
}
