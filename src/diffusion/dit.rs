// [WFGY] Zone: SAFE | λ: 0.25 | Fallbacks: 0 | Action: Diffusion Transformer (MMDiT / Flux.1 / SD 3.5) core architecture

pub mod blocks;
pub mod cogvideox;
pub mod embeddings;
pub mod flux;
pub mod minimax_h3;
pub mod pixart;
pub mod streamer;
pub mod z_image;

pub use blocks::{DoubleStreamBlock, SingleStreamBlock};
pub use cogvideox::{CogVideoXConfig, CogVideoXTransformer3D};
pub use embeddings::{apply_rope, create_flux_rope_embeddings, AdaLNZeroModulation, TimestepEmbedder};
pub use flux::{FluxConfig, FluxTransformer};
pub use minimax_h3::{MiniMaxH3Config, MiniMaxH3Transformer};
pub use pixart::{PixArtConfig, PixArtTransformer};
pub use streamer::SequentialBlockStreamer;
pub use z_image::{ZImageConfig, ZImageTransformer};

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
