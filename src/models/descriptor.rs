// [WFGY] Zone: SAFE | λ: 0.25 | Fallbacks: 0 | Action: Model descriptor — explicit checkpoint + VLM + VAE wiring per model family

use std::path::PathBuf;
use crate::models::config::Architecture;

/// Which text encoder to attach to a diffusion model (for families like Flux.2-Klein/Dev that do NOT
/// embed the conditioning text encoder). Self-contained models (Flux.1, SDXL) leave this `None` and
/// use their embedded encoders.
#[derive(Debug, Clone)]
pub enum TextEncoderSpec {
    /// Qwen3 encoder from a single `.safetensors` file or a shards directory.
    Qwen3 { path: PathBuf },
    /// Mistral-3 VLM from a directory of shards (Flux.2-Dev official).
    Mistral3 { dir: PathBuf },
    /// T5-XXL (rarely external — Flux.1 embeds it) from a safetensors file.
    T5 { path: PathBuf },
}

/// Explicit, per-model wiring: the DiT/UNet checkpoint plus the external VLM + VAE it needs
/// (if any). This is what a platform uses to *control* model loading (and unload on switch),
/// rather than relying on auto-detection of embedded encoders.
#[derive(Debug, Clone)]
pub struct ModelDescriptor {
    pub id: String,
    pub checkpoint: PathBuf,
    /// Optional hint; when `None` the architecture is sniffed from the checkpoint.
    pub family: Option<Architecture>,
    pub text_encoder: Option<TextEncoderSpec>,
    /// Path to a 32-channel Flux VAE (`flux2-vae.safetensors`) for Flux.2 families that don't embed
    /// one. `None` means the VAE is embedded (Flux.1) or handled by the SDXL pipeline.
    pub vae: Option<PathBuf>,
}

impl ModelDescriptor {
    /// A self-contained model (SDXL, Flux.1): no external VLM/VAE.
    pub fn standalone(id: impl Into<String>, checkpoint: impl Into<PathBuf>) -> Self {
        Self { id: id.into(), checkpoint: checkpoint.into(), family: None, text_encoder: None, vae: None }
    }

    /// A Flux.2-Klein/Dev model: attach the matching VLM and the shared 32-ch Flux VAE.
    pub fn flux2(
        id: impl Into<String>,
        checkpoint: impl Into<PathBuf>,
        text_encoder: TextEncoderSpec,
        vae: impl Into<PathBuf>,
    ) -> Self {
        Self {
            id: id.into(),
            checkpoint: checkpoint.into(),
            family: None,
            text_encoder: Some(text_encoder),
            vae: Some(vae.into()),
        }
    }
}
