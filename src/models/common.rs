// [WFGY] Zone: SAFE | λ: 0.25 | Fallbacks: 0 | Action: Common `transformers`-like model traits (port of HF API to Candle), object-safe

use candle_core::{DType, Device, Tensor};
use image::RgbImage;
use std::any::Any;
use std::path::Path;
use crate::error::Result;
use crate::traits::{DiffusionParams, Img2ImgParams, InpaintParams};

/// Progress callback (step_index, total_steps, latent). Used object-safe (boxed, not generic).
pub type ProgressFn<'a> = &'a mut (dyn FnMut(usize, usize, &Tensor) + 'a);


/// The broad category a loaded model belongs to. Mirrors the way HuggingFace groups
/// `AutoModelForCausalLM`, `DiffusionPipeline`, `AutoModelForImageClassification`, etc.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelKind {
    /// Autoregressive text (Qwen / Mistral as a CausalLM) — Milestone 16.
    Text,
    /// Text-encoder side of a diffused model (Qwen / Mistral / T5 / CLIP used for conditioning).
    TextEncoder,
    /// Text-to-image / img2img / inpaint diffusion model (Flux, SDXL, SD1.5).
    Diffusion,
    /// Could not be determined from the checkpoint/`config.json`.
    Unknown,
}

/// Everything a model agrees to expose (metadata only). Every concrete model implements this.
/// Object-safe: this is the type boxed as `Arc<Mutex<dyn AnyModel>>`.
pub trait AnyModel: Send + Sync {
    fn kind(&self) -> ModelKind;
    /// Stable family slug, e.g. "qwen3", "mistral3", "flux2-dev", "sdxl".
    fn family(&self) -> &str;
    fn id(&self) -> &str;
    fn device(&self) -> &Device;
    fn dtype(&self) -> DType;

    /// Upcast to `&dyn Any` so a caller can downcast to a concrete sub-trait
    /// (`ImageGenerationModel`, `EncodeModel`, ...). Implement each concrete model with `self`.
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// Downcast a boxed trait object held behind a lock to a concrete sub-trait.
///
/// The engine returns `Arc<Mutex<dyn AnyModel>>`; callers that want to drive a specific capability
/// (image vs text) downcast once. Returns `None` when the model is a different kind.
pub fn downcast_model<M: AnyModel + 'static>(m: &mut dyn AnyModel) -> Option<&mut M> {
    m.as_any_mut().downcast_mut::<M>()
}

/// Text-to-text generator (the CausalLM use case). Implemented by a decoder-only pipeline.
pub trait GenerationModel: AnyModel {
    fn generate(&mut self, prompt: &str, max_tokens: usize, temperature: f64) -> Result<String>;
}

/// Text-encoder (conditioning) use case: prompt -> embedding tensor.
pub trait EncodeModel: AnyModel {
    fn encode(&mut self, prompt: &str, max_len: usize) -> Result<Tensor>;
}

/// Image diffusion (T2I / img2img / inpaint). Object-safe: callbacks are boxed, not generic.
pub trait ImageGenerationModel: AnyModel {
    fn generate_t2i(&mut self, params: DiffusionParams, on_step: Option<ProgressFn>) -> Result<RgbImage>;
    fn generate_img2img(&mut self, params: Img2ImgParams, on_step: Option<ProgressFn>) -> Result<RgbImage>;
    fn generate_inpaint(&mut self, params: InpaintParams, on_step: Option<ProgressFn>) -> Result<RgbImage>;
    fn load_lora(&mut self, path: &Path, multiplier: f64) -> Result<()>;
    fn unload_lora(&mut self, id: &str) -> Result<()>;
    fn lora_ids(&self) -> Vec<String>;
}
