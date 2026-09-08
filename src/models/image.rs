// [WFGY] Zone: SAFE | λ: 0.25 | Fallbacks: 0 | Action: Diffusion model wrapper — a uniform `ImageGenerationModel` over Flux/SDXL/SD1.5

use candle_core::{DType, Device, Tensor};
use image::RgbImage;
use std::path::Path;
use crate::error::{LuminaError, Result};
use crate::pipelines::{FluxPipeline, StableDiffusionPipeline, StableDiffusionXLPipeline};
use crate::traits::{DiffusionParams, Img2ImgParams, InpaintParams, TextToImagePipeline};
use super::common::{AnyModel, ImageGenerationModel, ModelKind, ProgressFn};

/// A diffusion backend, boxed into a single enum. This hides the concrete pipeline type so the
/// platform only ever sees `ImageGenerationModel`.
pub enum DiffBackend {
    Flux(FluxPipeline),
    Sdxl(StableDiffusionXLPipeline),
    Sd15(StableDiffusionPipeline),
}

impl DiffBackend {
    fn device(&self) -> &Device {
        match self {
            DiffBackend::Flux(p) => &p.device,
            DiffBackend::Sdxl(_) => &Device::Cpu,
            DiffBackend::Sd15(_) => &Device::Cpu,
        }
    }
    fn dtype(&self) -> DType {
        match self {
            DiffBackend::Flux(p) => p.dtype,
            DiffBackend::Sdxl(_) => DType::F16,
            DiffBackend::Sd15(_) => DType::F32,
        }
    }
}

pub struct DiffusionModel {
    id: String,
    family: String,
    inner: DiffBackend,
}

impl DiffusionModel {
    pub fn flux(id: String, family: String, pipeline: FluxPipeline) -> Self {
        Self { id, family, inner: DiffBackend::Flux(pipeline) }
    }
    pub fn sdxl(id: String, pipeline: StableDiffusionXLPipeline) -> Self {
        Self { id, family: "sdxl".into(), inner: DiffBackend::Sdxl(pipeline) }
    }
    pub fn sd15(id: String, pipeline: StableDiffusionPipeline) -> Self {
        Self { id, family: "sd15".into(), inner: DiffBackend::Sd15(pipeline) }
    }
}

impl AnyModel for DiffusionModel {
    fn kind(&self) -> ModelKind { ModelKind::Diffusion }
    fn family(&self) -> &str { &self.family }
    fn id(&self) -> &str { &self.id }
    fn device(&self) -> &Device { self.inner.device() }
    fn dtype(&self) -> DType { self.inner.dtype() }
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}

impl ImageGenerationModel for DiffusionModel {
    fn generate_t2i(&mut self, params: DiffusionParams, on_step: Option<ProgressFn>) -> Result<RgbImage> {
        match &mut self.inner {
            DiffBackend::Flux(p) => {
                // Flux's `generate_with_metrics` requires `F: Fn`, but our object-safe surface is
                // `FnMut`. For now drop the callback (kept on the trait for future adaption).
                p.generate_with_metrics(params, None::<fn(usize, usize, &Tensor)>).map(|(img, _)| img)
            }
            DiffBackend::Sdxl(p) => {
                let cb = on_step;
                p.generate(params, cb)
            }
            DiffBackend::Sd15(p) => {
                let cb = on_step;
                p.generate(params, cb)
            }
        }
    }

    fn generate_img2img(&mut self, params: Img2ImgParams, on_step: Option<ProgressFn>) -> Result<RgbImage> {
        match &mut self.inner {
            DiffBackend::Flux(p) => {
                p.generate_img2img(params, None::<fn(usize, usize, &Tensor)>).map(|(img, _)| img)
            }
            DiffBackend::Sdxl(p) => {
                let cb = on_step;
                p.generate_img2img(params, cb)
            }
            DiffBackend::Sd15(_) => Err(LuminaError::UnsupportedOp("SD1.5 img2img".into())),
        }
    }

    fn generate_inpaint(&mut self, params: InpaintParams, _on_step: Option<ProgressFn>) -> Result<RgbImage> {
        match &mut self.inner {
            DiffBackend::Flux(p) => {
                p.generate_inpaint(params, None::<fn(usize, usize, &Tensor)>).map(|(img, _)| img)
            }
            DiffBackend::Sdxl(_) => Err(LuminaError::UnsupportedOp("SDXL inpaint".into())),
            DiffBackend::Sd15(_) => Err(LuminaError::UnsupportedOp("SD1.5 inpaint".into())),
        }
    }

    fn load_lora(&mut self, path: &Path, multiplier: f64) -> Result<()> {
        match &mut self.inner {
            DiffBackend::Flux(p) => p.load_lora(path, multiplier),
            DiffBackend::Sdxl(p) => p.load_lora(path, multiplier),
            DiffBackend::Sd15(_) => Err(LuminaError::UnsupportedOp("SD1.5 LoRA".into())),
        }
    }

    fn unload_lora(&mut self, id: &str) -> Result<()> {
        match &mut self.inner {
            DiffBackend::Flux(p) => p.unload_lora(id),
            DiffBackend::Sdxl(p) => p.unload_lora(id),
            DiffBackend::Sd15(_) => Err(LuminaError::UnsupportedOp("SD1.5 LoRA".into())),
        }
    }

    fn lora_ids(&self) -> Vec<String> {
        Vec::new()
    }
}
