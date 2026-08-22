// [WFGY] Zone: SAFE | λ: 0.25 | Fallbacks: 0 | Action: Diffusion Transformer (MMDiT / Flux.1 / SD 3.5) core architecture

pub mod blocks;
pub mod embeddings;
pub mod flux;

pub use blocks::{DoubleStreamBlock, SingleStreamBlock};
pub use embeddings::{apply_rope, create_rope_frequencies, AdaLNZeroModulation, TimestepEmbedder};
pub use flux::{FluxConfig, FluxTransformer};

use candle_core::{Result, Tensor};

/// General abstraction for Diffusion Transformers
pub trait DiffusionTransformer {
    fn forward(
        &self,
        latents: &Tensor,
        timestep: &Tensor,
        context: &Tensor,
        guidance: Option<&Tensor>,
    ) -> Result<Tensor>;
}
