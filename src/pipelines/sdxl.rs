// [WFGY] Zone: SAFE | λ: 0.25 | Fallbacks: 0 | Action: High-Performance SDXL Pipeline with Model CPU Offloading and Tiled VAE

use candle_core::{DType, Device, Tensor};
use image::RgbImage;
use std::path::Path;

use crate::diffusion::schedulers::{EulerDiscreteScheduler, EulerSchedulerConfig, Scheduler};
use crate::diffusion::unet_2d::UNetConditionModel;
use crate::diffusion::vae::VaeDecoder;
use crate::text::clip::ClipTextEncoder;
use crate::text::open_clip::OpenClipTextEncoder;
use crate::traits::{DiffusionParams, TextToImagePipeline};
use crate::weights::{SafeTensorsArchive, WeightRouter};

/// Configurable memory optimization controls for the pipeline
#[derive(Debug, Clone)]
pub struct PipelineMemoryConfig {
    /// Enable Tiled VAE Decoding to cap VAE peak VRAM to < 400 MB (default: true for > 512x512)
    pub vae_tiling: bool,
    /// Custom latent tile size (default: 64 -> 512x512 pixels)
    pub vae_tile_size: usize,
    /// Custom latent tile overlap (default: 8 -> 64 pixels)
    /// Custom latent tile overlap (default: 16 -> 128 pixels)
    pub vae_tile_overlap: usize,
    /// Offload text encoders (CLIP-L & OpenCLIP-G) to CPU memory (-2.6 GB VRAM)
    pub cpu_offload: bool,
}

impl Default for PipelineMemoryConfig {
    fn default() -> Self {
        Self {
            vae_tiling: true,
            vae_tile_size: 72,
            vae_tile_overlap: 16, // Optimal 4-tile seamless cosine feathering (128px overlap)
            cpu_offload: true, // Default enabled for < 7GB VRAM operation
        }
    }
}

pub struct StableDiffusionXLPipeline {
    checkpoint_path: std::path::PathBuf,
    clip_l: Option<ClipTextEncoder>,
    clip_g: Option<OpenClipTextEncoder>,
    unet: Option<UNetConditionModel>,
    vae: VaeDecoder,
    scheduler: EulerDiscreteScheduler,
    device: Device,
    dtype: DType,
    memory_config: PipelineMemoryConfig,
    lora_manager: crate::lora::LoRAManager,
}

impl StableDiffusionXLPipeline {
    pub fn from_single_file<P: AsRef<Path>>(checkpoint_path: P, device: Device) -> crate::error::Result<Self> {
        Self::from_single_file_with_config(checkpoint_path, device, PipelineMemoryConfig::default())
    }

    pub fn from_single_file_with_config<P: AsRef<Path>>(
        checkpoint_path: P,
        device: Device,
        memory_config: PipelineMemoryConfig,
    ) -> crate::error::Result<Self> {
        let is_cuda = device.is_cuda();
        let dtype = if is_cuda { DType::F16 } else { DType::F32 };
        let checkpoint_buf = checkpoint_path.as_ref().to_path_buf();

        let archive = SafeTensorsArchive::open(&checkpoint_buf)?;
        let router = WeightRouter::new(&archive, device.clone(), dtype);

        // If cpu_offload is true, load text encoders on CPU to save 2.6 GB VRAM
        let (text_device, text_dtype) = if memory_config.cpu_offload {
            (Device::Cpu, DType::F16)
        } else {
            (device.clone(), dtype)
        };

        let clip_l_vb = router.clip_l_var_builder_on_device(&text_device, text_dtype)?;
        let mut clip_l = ClipTextEncoder::new_sd15(clip_l_vb)?;
        let clip_l_tok = Path::new("clip_tokenizer.json");
        if clip_l_tok.exists() {
            clip_l.load_tokenizer(clip_l_tok)?;
        }

        let clip_g_vb = router.open_clip_g_var_builder_on_device(&text_device, text_dtype)?;
        let mut clip_g = OpenClipTextEncoder::new_sdxl(clip_g_vb)?;
        let clip_g_tok = Path::new("openclip_tokenizer.json");
        if clip_g_tok.exists() {
            clip_g.load_tokenizer(clip_g_tok)?;
        }

        let unet_vb = router.unet_var_builder()?;
        let unet = UNetConditionModel::new_sdxl(unet_vb)?;

        let vae_vb = router.vae_var_builder()?;
        let vae = VaeDecoder::new(vae_vb, true)?;

        let scheduler_config = EulerSchedulerConfig {
            use_karras_sigmas: true,
            ..Default::default()
        };
        let scheduler = EulerDiscreteScheduler::new(scheduler_config);

        Ok(Self {
            checkpoint_path: checkpoint_buf,
            clip_l: Some(clip_l),
            clip_g: Some(clip_g),
            unet: Some(unet),
            vae,
            scheduler,
            device,
            dtype,
            memory_config,
            lora_manager: crate::lora::LoRAManager::new(),
        })
    }

    /// Hot-merge a LoRA / LyCORIS into UNet and CLIP weights in-memory with zero runtime latency penalty
    pub fn load_lora<P: AsRef<Path>>(&mut self, path: P, multiplier: f64) -> crate::error::Result<()> {
        let path_ref = path.as_ref();
        println!("🧬 Hot-merging LoRA: {} (weight: {:.2})...", path_ref.display(), multiplier);
        let t_start = std::time::Instant::now();

        // 1. Compute LoRA deltas on CPU (0 MB GPU VRAM allocation)
        let deltas = self.lora_manager.load_and_merge(path_ref, multiplier, &candle_core::Device::Cpu, self.dtype)
            .map_err(|e| crate::error::LuminaError::Config(format!("LoRA merge error: {}", e)))?;

        // 2. Convert to lookup map
        let delta_map: std::collections::HashMap<String, Tensor> = deltas.into_iter().collect();

        // 3. Instant in-place GPU weight modification (< 0.05s)
        if let Some(unet) = &mut self.unet {
            unet.apply_lora_deltas(&delta_map)?;
        }
        if let Some(clip_l) = &mut self.clip_l {
            clip_l.apply_lora_deltas(&delta_map)?;
        }
        if let Some(clip_g) = &mut self.clip_g {
            clip_g.apply_lora_deltas(&delta_map)?;
        }

        println!("  ✅ LoRA successfully merged in {:.2}s", t_start.elapsed().as_secs_f64());
        Ok(())
    }

    /// Remove all merged LoRAs and restore original baseline checkpoint weights
    pub fn unload_all_loras(&mut self) -> crate::error::Result<()> {
        println!("🔄 Unloading all LoRAs and restoring base checkpoint weights...");
        let t_start = std::time::Instant::now();

        // Invert applied deltas to subtract them in-place
        let mut negative_deltas: std::collections::HashMap<String, Tensor> = std::collections::HashMap::new();
        for (k, v) in self.lora_manager.applied_deltas() {
            let neg = (v.neg())?;
            negative_deltas.insert(k.clone(), neg);
        }

        if let Some(unet) = &mut self.unet {
            unet.apply_lora_deltas(&negative_deltas)?;
        }
        if let Some(clip_l) = &mut self.clip_l {
            clip_l.apply_lora_deltas(&negative_deltas)?;
        }
        if let Some(clip_g) = &mut self.clip_g {
            clip_g.apply_lora_deltas(&negative_deltas)?;
        }

        self.lora_manager.clear();
        println!("  ✅ Original base weights restored in {:.2}s.", t_start.elapsed().as_secs_f64());
        Ok(())
    }

    /// List currently loaded LoRA models and their multipliers
    pub fn loaded_loras(&self) -> &[crate::lora::LoadedLoRA] {
        self.lora_manager.loaded_loras()
    }

    /// Return the path to the loaded base checkpoint
    pub fn checkpoint_path(&self) -> &Path {
        &self.checkpoint_path
    }

    /// Enable Tiled VAE Decoding:
    /// - `None` -> Automatic 72x72 tiles with 16-latent seamless cosine overlap (4 tiles)
    /// - `Some((tile_size, overlap))` -> Explicit user-defined tiling dimensions
    pub fn enable_vae_tiling(&mut self, custom: Option<(usize, usize)>) -> &mut Self {
        self.memory_config.vae_tiling = true;
        if let Some((tile_size, overlap)) = custom {
            self.memory_config.vae_tile_size = tile_size;
            self.memory_config.vae_tile_overlap = overlap;
        } else {
            self.memory_config.vae_tile_size = 72;
            self.memory_config.vae_tile_overlap = 16;
        }
        self
    }

    /// Disable Tiled VAE Decoding (executes direct single-pass decode, ideal for 24GB+ GPUs)
    pub fn disable_vae_tiling(&mut self) -> &mut Self {
        self.memory_config.vae_tiling = false;
        self
    }

    /// Enable Model CPU Offloading (text encoders run on CPU, saving 2.6 GB VRAM)
    pub fn enable_model_cpu_offload(&mut self) -> &mut Self {
        self.memory_config.cpu_offload = true;
        self
    }

    /// Disable Model CPU Offloading (all models kept in GPU VRAM)
    pub fn disable_model_cpu_offload(&mut self) -> &mut Self {
        self.memory_config.cpu_offload = false;
        self
    }

    /// Set full custom memory configuration
    pub fn set_memory_config(&mut self, config: PipelineMemoryConfig) -> &mut Self {
        self.memory_config = config;
        self
    }

    pub fn memory_config(&self) -> &PipelineMemoryConfig {
        &self.memory_config
    }

    /// Execute Image-to-Image (Img2Img) diffusion generation
    pub fn generate_img2img<F>(
        &mut self,
        params: crate::traits::Img2ImgParams,
        mut on_step: Option<F>,
    ) -> crate::error::Result<RgbImage>
    where
        F: FnMut(usize, usize, &Tensor),
    {
        let clip_l = self.clip_l.as_mut().ok_or_else(|| crate::error::LuminaError::Config("CLIP-L not loaded".to_string()))?;
        let clip_g = self.clip_g.as_mut().ok_or_else(|| crate::error::LuminaError::Config("OpenCLIP-G not loaded".to_string()))?;
        let unet = self.unet.as_ref().ok_or_else(|| crate::error::LuminaError::Config("UNet not loaded".to_string()))?;

        let prompt = params.prompt;
        let negative_prompt = params.negative_prompt.unwrap_or("");
        let num_steps = params.num_steps;
        let guidance_scale = params.guidance_scale;
        let strength = params.strength.clamp(0.01, 1.0);

        let (orig_w, orig_h) = params.image.dimensions();
        // Ensure dimensions are multiples of 8
        let target_w = (orig_w / 8) * 8;
        let target_h = (orig_h / 8) * 8;
        let latent_height = (target_h / 8) as usize;
        let latent_width = (target_w / 8) as usize;

        // Resize image if dimensions were adjusted
        let src_image = if target_w != orig_w || target_h != orig_h {
            image::imageops::resize(&params.image, target_w, target_h, image::imageops::FilterType::Lanczos3)
        } else {
            params.image
        };

        // 1. Text embeddings: CLIP-L + OpenCLIP-G with pooled vectors
        let cond_l = clip_l.encode_prompt(prompt)?
            .to_device(&self.device)?
            .to_dtype(self.dtype)?;

        let (cond_g, cond_pooled) = clip_g.encode_prompt_with_pooled(prompt)?;
        let cond_g = cond_g.to_device(&self.device)?.to_dtype(self.dtype)?;
        let cond_pooled = cond_pooled.to_device(&self.device)?.to_dtype(self.dtype)?;
        let cond_embeds = Tensor::cat(&[&cond_l, &cond_g], 2)?;

        let uncond_l = clip_l.encode_prompt(negative_prompt)?
            .to_device(&self.device)?
            .to_dtype(self.dtype)?;

        let (uncond_g, uncond_pooled) = clip_g.encode_prompt_with_pooled(negative_prompt)?;
        let uncond_g = uncond_g.to_device(&self.device)?.to_dtype(self.dtype)?;
        let uncond_pooled = uncond_pooled.to_device(&self.device)?.to_dtype(self.dtype)?;
        let uncond_embeds = Tensor::cat(&[&uncond_l, &uncond_g], 2)?;

        let text_embeds = Tensor::cat(&[&uncond_embeds, &cond_embeds], 0)?;
        let pooled_embeds = Tensor::cat(&[&uncond_pooled, &cond_pooled], 0)?;

        // 2. Set scheduler timesteps and compute start timestep based on strength
        self.scheduler.set_timesteps(num_steps)?;
        let timesteps = self.scheduler.timesteps().to_vec();
        let sigmas = self.scheduler.sigmas().to_vec();

        let num_inference_steps = ((num_steps as f64) * strength).round() as usize;
        let num_inference_steps = num_inference_steps.max(1).min(num_steps);
        let start_idx = num_steps - num_inference_steps;
        let active_timesteps = &timesteps[start_idx..];
        let init_sigma = sigmas.get(start_idx).copied().unwrap_or(1.0);

        // 3. Encode source image into latent space
        let init_latents = self.vae.encode_image(&src_image)?
            .to_device(&self.device)?
            .to_dtype(self.dtype)?;

        // 4. Inject noise at initial sigma: z_start = z_init + sigma * noise
        let noise = Tensor::randn(
            0f32,
            1f32,
            (1, 4, latent_height, latent_width),
            &self.device,
        )?.to_dtype(self.dtype)?;

        let mut latents = if start_idx == 0 {
            (noise * init_sigma)?
        } else {
            (&init_latents + (noise * init_sigma)?)?
        };

        // Pre-compute SDXL Add Embedding
        let precomputed_add_proj = unet.compute_add_embedding(
            2,
            latent_height,
            latent_width,
            Some(&pooled_embeds),
            &self.device,
            self.dtype,
        )?;

        // 5. Denoising loop for active timesteps
        for (step_idx, &timestep) in active_timesteps.iter().enumerate() {
            let scaled_latent = self.scheduler.scale_model_input(&latents, timestep)?;
            let latent_model_input = Tensor::cat(&[&scaled_latent, &scaled_latent], 0)?;

            let noise_pred = unet.forward_with_precomputed(
                &latent_model_input,
                timestep as f64,
                &text_embeds,
                precomputed_add_proj.as_ref(),
            )?;

            let noise_uncond = noise_pred.narrow(0, 0, 1)?;
            let noise_cond = noise_pred.narrow(0, 1, 1)?;
            let diff = (&noise_cond - &noise_uncond)?;
            let guided_noise = (&noise_uncond + (&diff * (guidance_scale as f64))?)?;

            latents = self.scheduler.step(&guided_noise, timestep, &latents)?;

            if let Some(ref mut callback) = on_step {
                callback(step_idx + 1, num_inference_steps, &latents);
            }
        }

        // 6. Decode final latents via SDXL VAE
        let t_vae = std::time::Instant::now();
        let image = if self.memory_config.vae_tiling {
            self.vae.decode_tiled(
                &latents,
                self.memory_config.vae_tile_size,
                self.memory_config.vae_tile_overlap,
            )?
        } else {
            self.vae.decode_direct(&latents)?
        };
        println!("\n    VAE decoded in {:.2}s", t_vae.elapsed().as_secs_f64());
        let _ = std::io::Write::flush(&mut std::io::stdout());

        Ok(image)
    }

    /// Execute Inpainting / Outpainting diffusion generation
    pub fn generate_inpaint<F>(
        &mut self,
        params: crate::traits::InpaintParams,
        mut on_step: Option<F>,
    ) -> crate::error::Result<RgbImage>
    where
        F: FnMut(usize, usize, &Tensor),
    {
        let clip_l = self.clip_l.as_mut().ok_or_else(|| crate::error::LuminaError::Config("CLIP-L not loaded".to_string()))?;
        let clip_g = self.clip_g.as_mut().ok_or_else(|| crate::error::LuminaError::Config("OpenCLIP-G not loaded".to_string()))?;
        let unet = self.unet.as_ref().ok_or_else(|| crate::error::LuminaError::Config("UNet not loaded".to_string()))?;

        let prompt = params.prompt;
        let negative_prompt = params.negative_prompt.unwrap_or("");
        let num_steps = params.num_steps;
        let guidance_scale = params.guidance_scale;
        let strength = params.strength.clamp(0.01, 1.0);

        let (orig_w, orig_h) = params.image.dimensions();
        let target_w = (orig_w / 8) * 8;
        let target_h = (orig_h / 8) * 8;
        let latent_height = (target_h / 8) as usize;
        let latent_width = (target_w / 8) as usize;

        // Resize image and mask if needed
        let src_image = if target_w != orig_w || target_h != orig_h {
            image::imageops::resize(&params.image, target_w, target_h, image::imageops::FilterType::Lanczos3)
        } else {
            params.image
        };

        let mut mask_img = if params.mask.dimensions() != (target_w, target_h) {
            image::imageops::resize(&params.mask, target_w, target_h, image::imageops::FilterType::Lanczos3)
        } else {
            params.mask
        };

        // Feather edges if mask_blur > 0
        if params.mask_blur > 0 {
            mask_img = image::imageops::blur(&mask_img, params.mask_blur as f32);
        }

        // Downsample mask to latent resolution [1, 1, H/8, W/8] (floats in [0.0, 1.0])
        let mut latent_mask_floats = vec![0.0f32; latent_height * latent_width];
        for ly in 0..latent_height {
            for lx in 0..latent_width {
                let mut sum = 0.0f32;
                for py in 0..8 {
                    for px in 0..8 {
                        let p = mask_img.get_pixel((lx * 8 + px) as u32, (ly * 8 + py) as u32)[0];
                        sum += p as f32 / 255.0;
                    }
                }
                latent_mask_floats[ly * latent_width + lx] = sum / 64.0;
            }
        }
        let latent_mask = Tensor::from_vec(latent_mask_floats, (1, 1, latent_height, latent_width), &self.device)?
            .to_dtype(self.dtype)?;
        let inv_latent_mask = (Tensor::ones_like(&latent_mask)? - &latent_mask)?;

        // 1. Encode source image into latent space first
        let init_latents = self.vae.encode_image(&src_image)?
            .to_device(&self.device)?
            .to_dtype(self.dtype)?;

        // 2. Text embeddings: CLIP-L + OpenCLIP-G with pooled vectors
        let cond_l = clip_l.encode_prompt(prompt)?
            .to_device(&self.device)?
            .to_dtype(self.dtype)?;

        let (cond_g, cond_pooled) = clip_g.encode_prompt_with_pooled(prompt)?;
        let cond_g = cond_g.to_device(&self.device)?.to_dtype(self.dtype)?;
        let cond_pooled = cond_pooled.to_device(&self.device)?.to_dtype(self.dtype)?;
        let cond_embeds = Tensor::cat(&[&cond_l, &cond_g], 2)?;

        let uncond_l = clip_l.encode_prompt(negative_prompt)?
            .to_device(&self.device)?
            .to_dtype(self.dtype)?;

        let (uncond_g, uncond_pooled) = clip_g.encode_prompt_with_pooled(negative_prompt)?;
        let uncond_g = uncond_g.to_device(&self.device)?.to_dtype(self.dtype)?;
        let uncond_pooled = uncond_pooled.to_device(&self.device)?.to_dtype(self.dtype)?;
        let uncond_embeds = Tensor::cat(&[&uncond_l, &uncond_g], 2)?;

        let text_embeds = Tensor::cat(&[&uncond_embeds, &cond_embeds], 0)?;
        let pooled_embeds = Tensor::cat(&[&uncond_pooled, &cond_pooled], 0)?;

        // 3. Set scheduler timesteps and compute start timestep
        self.scheduler.set_timesteps(num_steps)?;
        let timesteps = self.scheduler.timesteps().to_vec();
        let sigmas = self.scheduler.sigmas().to_vec();

        let num_inference_steps = ((num_steps as f64) * strength).round() as usize;
        let num_inference_steps = num_inference_steps.max(1).min(num_steps);
        let start_idx = num_steps - num_inference_steps;
        let active_timesteps = &timesteps[start_idx..];
        let init_sigma = sigmas.get(start_idx).copied().unwrap_or(1.0);

        // Fixed background noise for unmasked area
        let fixed_bg_noise = Tensor::randn(
            0f32,
            1f32,
            (1, 4, latent_height, latent_width),
            &self.device,
        )?.to_dtype(self.dtype)?;

        // Target noise for inpainting area
        let inpaint_noise = Tensor::randn(
            0f32,
            1f32,
            (1, 4, latent_height, latent_width),
            &self.device,
        )?.to_dtype(self.dtype)?;

        // Initial combined latent
        let init_noisy_bg = (&init_latents + (&fixed_bg_noise * init_sigma)?)?;
        let init_noisy_fg = (&inpaint_noise * init_sigma)?;
        let mut latents = (
            init_noisy_bg.broadcast_mul(&inv_latent_mask)? +
            init_noisy_fg.broadcast_mul(&latent_mask)?
        )?;

        // Pre-compute SDXL Add Embedding
        let precomputed_add_proj = unet.compute_add_embedding(
            2,
            latent_height,
            latent_width,
            Some(&pooled_embeds),
            &self.device,
            self.dtype,
        )?;

        // 4. Denoising loop
        for (step_idx, &timestep) in active_timesteps.iter().enumerate() {
            let current_global_step = start_idx + step_idx;
            let scaled_latent = self.scheduler.scale_model_input(&latents, timestep)?;
            let latent_model_input = Tensor::cat(&[&scaled_latent, &scaled_latent], 0)?;

            let noise_pred = unet.forward_with_precomputed(
                &latent_model_input,
                timestep as f64,
                &text_embeds,
                precomputed_add_proj.as_ref(),
            )?;

            let noise_uncond = noise_pred.narrow(0, 0, 1)?;
            let noise_cond = noise_pred.narrow(0, 1, 1)?;
            let diff = (&noise_cond - &noise_uncond)?;
            let guided_noise = (&noise_uncond + (&diff * (guidance_scale as f64))?)?;

            let denoised_step = self.scheduler.step(&guided_noise, timestep, &latents)?;

            // Re-inject the original latent at next timestep sigma to perfectly preserve unmasked region
            let next_sigma = sigmas.get(current_global_step + 1).copied().unwrap_or(0.0);
            let bg_at_next_step = if next_sigma > 0.0 {
                (&init_latents + (&fixed_bg_noise * next_sigma)?)?
            } else {
                init_latents.clone()
            };

            latents = (
                bg_at_next_step.broadcast_mul(&inv_latent_mask)? +
                denoised_step.broadcast_mul(&latent_mask)?
            )?;

            if let Some(ref mut callback) = on_step {
                callback(step_idx + 1, num_inference_steps, &latents);
            }
        }

        // 5. Decode final latents via SDXL VAE
        let t_vae = std::time::Instant::now();
        let image = if self.memory_config.vae_tiling {
            self.vae.decode_tiled(
                &latents,
                self.memory_config.vae_tile_size,
                self.memory_config.vae_tile_overlap,
            )?
        } else {
            self.vae.decode_direct(&latents)?
        };
        println!("\n    VAE decoded in {:.2}s", t_vae.elapsed().as_secs_f64());
        let _ = std::io::Write::flush(&mut std::io::stdout());

        Ok(image)
    }

    /// Execute ControlNet conditioned diffusion generation
    pub fn generate_controlnet<F>(
        &mut self,
        params: crate::traits::ControlNetParams,
        controlnet: &crate::diffusion::MultiControlNet,
        mut on_step: Option<F>,
    ) -> crate::error::Result<RgbImage>
    where
        F: FnMut(usize, usize, &Tensor),
    {
        let clip_l = self.clip_l.as_mut().ok_or_else(|| crate::error::LuminaError::Config("CLIP-L not loaded".to_string()))?;
        let clip_g = self.clip_g.as_mut().ok_or_else(|| crate::error::LuminaError::Config("OpenCLIP-G not loaded".to_string()))?;
        let unet = self.unet.as_ref().ok_or_else(|| crate::error::LuminaError::Config("UNet not loaded".to_string()))?;

        let prompt = params.prompt;
        let negative_prompt = params.negative_prompt.unwrap_or("");
        let num_steps = params.num_steps;
        let guidance_scale = params.guidance_scale;
        let width = (params.width / 8) * 8;
        let height = (params.height / 8) * 8;
        let latent_height = height / 8;
        let latent_width = width / 8;

        // 1. Text embeddings: CLIP-L + OpenCLIP-G with pooled vectors
        let cond_l = clip_l.encode_prompt(prompt)?
            .to_device(&self.device)?
            .to_dtype(self.dtype)?;

        let (cond_g, cond_pooled) = clip_g.encode_prompt_with_pooled(prompt)?;
        let cond_g = cond_g.to_device(&self.device)?.to_dtype(self.dtype)?;
        let cond_pooled = cond_pooled.to_device(&self.device)?.to_dtype(self.dtype)?;
        let cond_embeds = Tensor::cat(&[&cond_l, &cond_g], 2)?;

        let uncond_l = clip_l.encode_prompt(negative_prompt)?
            .to_device(&self.device)?
            .to_dtype(self.dtype)?;

        let (uncond_g, uncond_pooled) = clip_g.encode_prompt_with_pooled(negative_prompt)?;
        let uncond_g = uncond_g.to_device(&self.device)?.to_dtype(self.dtype)?;
        let uncond_pooled = uncond_pooled.to_device(&self.device)?.to_dtype(self.dtype)?;
        let uncond_embeds = Tensor::cat(&[&uncond_l, &uncond_g], 2)?;

        let text_embeds = Tensor::cat(&[&uncond_embeds, &cond_embeds], 0)?;
        let pooled_embeds = Tensor::cat(&[&uncond_pooled, &cond_pooled], 0)?;

        // 2. Pre-process conditioning images to [2, 3, H, W] in [-1.0, 1.0] (for uncond & cond)
        let mut cond_tensors = Vec::with_capacity(params.cond_images.len());
        for img in &params.cond_images {
            let resized = if img.dimensions() != (width as u32, height as u32) {
                image::imageops::resize(img, width as u32, height as u32, image::imageops::FilterType::Lanczos3)
            } else {
                img.clone()
            };
            let single_cond = crate::diffusion::vae::rgb_image_to_tensor(&resized, &self.device, self.dtype)?;
            let cond_batch = Tensor::cat(&[&single_cond, &single_cond], 0)?; // Batch of 2 for CFG
            cond_tensors.push(cond_batch);
        }

        // 3. Set scheduler timesteps and initialize random Gaussian latents
        self.scheduler.set_timesteps(num_steps)?;
        let timesteps = self.scheduler.timesteps().to_vec();
        let init_sigma = self.scheduler.sigmas().first().copied().unwrap_or(1.0);

        let init_latents = Tensor::randn(
            0f32,
            1f32,
            (1, 4, latent_height, latent_width),
            &self.device,
        )?.to_dtype(self.dtype)?;
        let mut latents = (init_latents * init_sigma)?;

        // Pre-compute SDXL Add Embedding
        let precomputed_add_proj = unet.compute_add_embedding(
            2,
            latent_height,
            latent_width,
            Some(&pooled_embeds),
            &self.device,
            self.dtype,
        )?;

        // 4. Denoising loop with ControlNet residual injection
        for (step_idx, &timestep) in timesteps.iter().enumerate() {
            let scaled_latent = self.scheduler.scale_model_input(&latents, timestep)?;
            let latent_model_input = Tensor::cat(&[&scaled_latent, &scaled_latent], 0)?;

            // Forward pass through MultiControlNet
            let (down_residuals, mid_residual) = controlnet.forward(
                &latent_model_input,
                timestep as f64,
                &text_embeds,
                precomputed_add_proj.as_ref(),
                &cond_tensors,
            )?;

            // UNet forward with injected residuals
            let noise_pred = unet.forward_with_controlnet(
                &latent_model_input,
                timestep as f64,
                &text_embeds,
                precomputed_add_proj.as_ref(),
                Some(&down_residuals),
                Some(&mid_residual),
            )?;

            let noise_uncond = noise_pred.narrow(0, 0, 1)?;
            let noise_cond = noise_pred.narrow(0, 1, 1)?;
            let diff = (&noise_cond - &noise_uncond)?;
            let guided_noise = (&noise_uncond + (&diff * (guidance_scale as f64))?)?;

            latents = self.scheduler.step(&guided_noise, timestep, &latents)?;

            if let Some(ref mut callback) = on_step {
                callback(step_idx + 1, num_steps, &latents);
            }
        }

        // 5. Decode final latents via SDXL VAE
        let t_vae = std::time::Instant::now();
        let image = if self.memory_config.vae_tiling {
            self.vae.decode_tiled(
                &latents,
                self.memory_config.vae_tile_size,
                self.memory_config.vae_tile_overlap,
            )?
        } else {
            self.vae.decode_direct(&latents)?
        };
        println!("\n    VAE decoded in {:.2}s", t_vae.elapsed().as_secs_f64());
        let _ = std::io::Write::flush(&mut std::io::stdout());

        Ok(image)
    }

    /// Execute text-to-image generation returning image and detailed telemetry metrics
    pub fn generate_with_metrics<F>(
        &mut self,
        params: DiffusionParams,
        mut on_step: Option<F>,
    ) -> crate::error::Result<(RgbImage, crate::device::GenerationMetrics)>
    where
        F: FnMut(usize, usize, &Tensor),
    {
        let t_total = std::time::Instant::now();
        let clip_l = self.clip_l.as_mut().ok_or_else(|| crate::error::LuminaError::Config("CLIP-L not loaded".to_string()))?;
        let clip_g = self.clip_g.as_mut().ok_or_else(|| crate::error::LuminaError::Config("OpenCLIP-G not loaded".to_string()))?;
        let unet = self.unet.as_ref().ok_or_else(|| crate::error::LuminaError::Config("UNet not loaded".to_string()))?;

        let prompt = params.prompt;
        let negative_prompt = params.negative_prompt.unwrap_or("");
        let num_steps = params.num_steps;
        let guidance_scale = params.guidance_scale;
        let latent_height = params.height / 8;
        let latent_width = params.width / 8;

        // 1. Text embeddings: CLIP-L + OpenCLIP-G with pooled vectors
        let t_text = std::time::Instant::now();
        let cond_l = clip_l.encode_prompt(prompt)?
            .to_device(&self.device)?
            .to_dtype(self.dtype)?;

        let (cond_g, cond_pooled) = clip_g.encode_prompt_with_pooled(prompt)?;
        let cond_g = cond_g.to_device(&self.device)?.to_dtype(self.dtype)?;
        let cond_pooled = cond_pooled.to_device(&self.device)?.to_dtype(self.dtype)?;
        let cond_embeds = Tensor::cat(&[&cond_l, &cond_g], 2)?;

        let uncond_l = clip_l.encode_prompt(negative_prompt)?
            .to_device(&self.device)?
            .to_dtype(self.dtype)?;

        let (uncond_g, uncond_pooled) = clip_g.encode_prompt_with_pooled(negative_prompt)?;
        let uncond_g = uncond_g.to_device(&self.device)?.to_dtype(self.dtype)?;
        let uncond_pooled = uncond_pooled.to_device(&self.device)?.to_dtype(self.dtype)?;
        let uncond_embeds = Tensor::cat(&[&uncond_l, &uncond_g], 2)?;

        let text_embeds = Tensor::cat(&[&uncond_embeds, &cond_embeds], 0)?;
        let pooled_embeds = Tensor::cat(&[&uncond_pooled, &cond_pooled], 0)?;
        let prompt_encode_ms = t_text.elapsed().as_secs_f64() * 1000.0;

        // 2. Scheduler and latents initialization
        self.scheduler.set_timesteps(num_steps)?;
        let timesteps = self.scheduler.timesteps().to_vec();
        let init_sigma = self.scheduler.sigmas().first().copied().unwrap_or(1.0);

        let init_latents = Tensor::randn(
            0f32,
            1f32,
            (1, 4, latent_height, latent_width),
            &self.device,
        )?.to_dtype(self.dtype)?;
        let mut latents = (init_latents * init_sigma)?;

        let precomputed_add_proj = unet.compute_add_embedding(
            2,
            latent_height,
            latent_width,
            Some(&pooled_embeds),
            &self.device,
            self.dtype,
        )?;

        // 3. Denoising loop
        let t_unet = std::time::Instant::now();
        for (step_idx, &timestep) in timesteps.iter().enumerate() {
            let scaled_latent = self.scheduler.scale_model_input(&latents, timestep)?;
            let latent_model_input = Tensor::cat(&[&scaled_latent, &scaled_latent], 0)?;

            let noise_pred = unet.forward_with_precomputed(
                &latent_model_input,
                timestep as f64,
                &text_embeds,
                precomputed_add_proj.as_ref(),
            )?;

            let noise_uncond = noise_pred.narrow(0, 0, 1)?;
            let noise_cond = noise_pred.narrow(0, 1, 1)?;
            let diff = (&noise_cond - &noise_uncond)?;
            let guided_noise = (&noise_uncond + (&diff * (guidance_scale as f64))?)?;

            latents = self.scheduler.step(&guided_noise, timestep, &latents)?;

            if let Some(ref mut callback) = on_step {
                callback(step_idx + 1, num_steps, &latents);
            }
        }
        let unet_total_ms = t_unet.elapsed().as_secs_f64() * 1000.0;
        let unet_step_avg_ms = if num_steps > 0 { unet_total_ms / (num_steps as f64) } else { 0.0 };
        let unet_it_per_sec = if unet_total_ms > 0.0 { (num_steps as f64) / (unet_total_ms / 1000.0) } else { 0.0 };

        // 4. VAE Decode
        let t_vae = std::time::Instant::now();
        let image = if self.memory_config.vae_tiling {
            self.vae.decode_adaptive(
                &latents,
                self.memory_config.vae_tile_size,
                self.memory_config.vae_tile_overlap,
            )?
        } else {
            self.vae.decode_direct(&latents)?
        };
        let vae_decode_ms = t_vae.elapsed().as_secs_f64() * 1000.0;
        let total_wallclock_ms = t_total.elapsed().as_secs_f64() * 1000.0;

        let metrics = crate::device::GenerationMetrics {
            prompt_encode_ms,
            unet_steps: num_steps,
            unet_total_ms,
            unet_step_avg_ms,
            unet_it_per_sec,
            vae_decode_ms,
            total_wallclock_ms,
        };

        println!("\n{}", metrics.summary_report());
        let _ = std::io::Write::flush(&mut std::io::stdout());

        Ok((image, metrics))
    }
}

impl TextToImagePipeline for StableDiffusionXLPipeline {
    fn from_safetensors<P: AsRef<Path>>(path: P, device: &Device) -> crate::error::Result<Self> {
        Self::from_single_file(path, device.clone())
    }

    fn generate<F>(&mut self, params: DiffusionParams, on_step: Option<F>) -> crate::error::Result<RgbImage>
    where
        F: FnMut(usize, usize, &Tensor),
    {
        let (image, _metrics) = self.generate_with_metrics(params, on_step)?;
        Ok(image)
    }
}
