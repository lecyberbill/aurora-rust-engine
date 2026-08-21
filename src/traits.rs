// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: High-level pipeline traits and common parameter abstractions

use candle_core::{Device, Tensor};
use image::RgbImage;
use std::path::Path;
use crate::error::Result;

/// Common parameters for diffusion text-to-image pipelines.
#[derive(Debug, Clone)]
pub struct DiffusionParams<'a> {
    pub prompt: &'a str,
    pub negative_prompt: Option<&'a str>,
    pub num_steps: usize,
    pub guidance_scale: f64,
    pub width: usize,
    pub height: usize,
    pub seed: u64,
}

impl<'a> Default for DiffusionParams<'a> {
    fn default() -> Self {
        Self {
            prompt: "",
            negative_prompt: None,
            num_steps: 25,
            guidance_scale: 7.5,
            width: 512,
            height: 512,
            seed: 42,
        }
    }
}

/// Parameters for Image-to-Image (Img2Img) generation.
#[derive(Debug, Clone)]
pub struct Img2ImgParams<'a> {
    pub prompt: &'a str,
    pub negative_prompt: Option<&'a str>,
    pub image: RgbImage,
    pub strength: f64,
    pub num_steps: usize,
    pub guidance_scale: f64,
    pub seed: u64,
}

impl<'a> Img2ImgParams<'a> {
    pub fn new(prompt: &'a str, image: RgbImage) -> Self {
        Self {
            prompt,
            negative_prompt: None,
            image,
            strength: 0.60,
            num_steps: 30,
            guidance_scale: 6.0,
            seed: 42,
        }
    }
}

/// Unified trait for full-pipeline Text-to-Image models (e.g. SD 1.5, SDXL, Flux).
pub trait TextToImagePipeline {
    /// Load weights directly from a monolithic or multi-part SafeTensors path.
    fn from_safetensors<P: AsRef<Path>>(path: P, device: &Device) -> Result<Self>
    where
        Self: Sized;

    /// Execute denoising loop with an optional per-step latent inspection callback.
    /// The callback receives `(step_index, total_steps, &latent_tensor)`.
    fn generate<F>(&mut self, params: DiffusionParams, on_step: Option<F>) -> Result<RgbImage>
    where
        F: FnMut(usize, usize, &Tensor);
}

/// Unified trait for Text-to-Text / Autoregressive LLM generation pipelines.
pub trait TextGenerationPipeline {
    fn generate(&mut self, prompt: &str, max_tokens: usize, temperature: f64) -> Result<String>;
}
