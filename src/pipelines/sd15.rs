// [WFGY] Zone: SAFE | λ: 0.30 | Fallbacks: 0 | Action: StableDiffusion 1.5 end-to-end inference pipeline

use candle_core::{DType, Device, Tensor};
use image::RgbImage;
use std::path::Path;
use tracing::info;

use crate::diffusion::schedulers::{EulerDiscreteScheduler, EulerSchedulerConfig, Scheduler};
use crate::diffusion::unet_2d::UNetConditionModel;
use crate::diffusion::vae::VaeDecoder;
use crate::error::Result;
use crate::text::clip::ClipTextEncoder;
use crate::traits::{DiffusionParams, TextToImagePipeline};
use crate::weights::{SafeTensorsArchive, WeightRouter};

pub struct StableDiffusionPipeline {
    text_encoder: ClipTextEncoder,
    unet: UNetConditionModel,
    vae: VaeDecoder,
    scheduler: Box<dyn Scheduler>,
    device: Device,
    dtype: DType,
}

impl StableDiffusionPipeline {
    pub fn new(
        text_encoder: ClipTextEncoder,
        unet: UNetConditionModel,
        vae: VaeDecoder,
        scheduler: Box<dyn Scheduler>,
        device: Device,
        dtype: DType,
    ) -> Self {
        Self {
            text_encoder,
            unet,
            vae,
            scheduler,
            device,
            dtype,
        }
    }

    /// Set a custom scheduler (e.g. DDIMScheduler, DPMSolver, etc.)
    pub fn set_scheduler(&mut self, scheduler: Box<dyn Scheduler>) {
        self.scheduler = scheduler;
    }
}

impl TextToImagePipeline for StableDiffusionPipeline {
    fn from_safetensors<P: AsRef<Path>>(path: P, device: &Device) -> Result<Self> {
        let dtype = DType::F32;
        let archive = SafeTensorsArchive::open(path)?;
        let router = WeightRouter::new(&archive, device.clone(), dtype);

        info!("Loading SD 1.5 UNet weights...");
        let unet_vb = router.var_builder_for_prefix("model.diffusion_model.", &["unet."])?;
        let unet = UNetConditionModel::new_sd15(unet_vb)?;

        info!("Loading SD 1.5 VAE weights...");
        let vae_vb = router.var_builder_for_prefix("first_stage_model.", &["vae."])?;
        let vae = VaeDecoder::new(vae_vb, false)?;

        info!("Loading SD 1.5 CLIP Text Encoder weights...");
        let text_vb = router.var_builder_for_prefix(
            "cond_stage_model.transformer.",
            &["conditioner.embedders.0.transformer.", "text_encoder."],
        )?;
        let text_encoder = ClipTextEncoder::new_sd15(text_vb)?;

        let scheduler = Box::new(EulerDiscreteScheduler::new(EulerSchedulerConfig::default()));

        Ok(Self {
            text_encoder,
            unet,
            vae,
            scheduler,
            device: device.clone(),
            dtype,
        })
    }

    fn generate<F>(&mut self, params: DiffusionParams, mut on_step: Option<F>) -> Result<RgbImage>
    where
        F: FnMut(usize, usize, &Tensor),
    {
        let prompt = params.prompt;
        let negative_prompt = params.negative_prompt.unwrap_or("");
        let num_steps = params.num_steps;
        let guidance_scale = params.guidance_scale;
        let latent_height = params.height / 8;
        let latent_width = params.width / 8;

        // 1. Text embeddings: conditional & unconditional
        let cond_embeds = self.text_encoder.encode_prompt(prompt)?;
        let uncond_embeds = self.text_encoder.encode_prompt(negative_prompt)?;

        // Concatenate for batched Classifier-Free Guidance [2, 77, 768]
        let text_embeds = Tensor::cat(&[&uncond_embeds, &cond_embeds], 0)?;

        // 2. Initialize latent Gaussian noise [1, 4, H/8, W/8]
        let mut latents = Tensor::randn(
            0f32,
            1f32,
            (1, 4, latent_height, latent_width),
            &self.device,
        )?.to_dtype(self.dtype)?;

        // 3. Initialize scheduler schedule
        self.scheduler.set_timesteps(num_steps)?;
        let timesteps = self.scheduler.timesteps().to_vec();

        // 4. Denoising loop
        for (step_idx, &timestep) in timesteps.iter().enumerate() {
            // Scale input latents according to scheduler
            let scaled_latent = self.scheduler.scale_model_input(&latents, timestep)?;
            let latent_model_input = Tensor::cat(&[&scaled_latent, &scaled_latent], 0)?;

            // Forward through UNet
            let noise_pred = self.unet.forward(
                &latent_model_input,
                timestep as f64,
                &text_embeds,
                None,
            )?;

            // Apply Classifier-Free Guidance (CFG)
            // noise_pred = uncond + guidance_scale * (cond - uncond)
            let noise_uncond = noise_pred.narrow(0, 0, 1)?;
            let noise_cond = noise_pred.narrow(0, 1, 1)?;
            let guided_noise = noise_uncond.add(&noise_cond.sub(&noise_uncond)?.affine(guidance_scale, 0.0)?)?;

            // Scheduler step
            latents = self.scheduler.step(&guided_noise, timestep, &latents)?;

            // Optional latent preview callback
            if let Some(ref mut callback) = on_step {
                callback(step_idx + 1, num_steps, &latents);
            }
        }

        // 5. Decode final latents to RGB Image via VAE
        info!("Decoding latents to RGB image...");
        let image = self.vae.decode_to_image(&latents)?;
        Ok(image)
    }
}
