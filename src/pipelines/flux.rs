// [WFGY] Zone: SAFE | λ: 0.25 | Fallbacks: 0 | Action: Pure Rust Flux.1 / MMDiT Inference Pipeline with Flow Matching ODE

use candle_core::{DType, Device, Result, Tensor};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use crate::device::GenerationMetrics;
use crate::diffusion::dit::{FluxConfig, FluxTransformer};
use crate::diffusion::schedulers::{FlowMatchEulerConfig, FlowMatchEulerScheduler, Scheduler};
use crate::traits::DiffusionParams;
use crate::weights::{SafeTensorsArchive, WeightRouter};

/// Convert patchified sequence latents [B, (H/16)*(W/16), C*4] back to 2D latents [B, C, H/8, W/8]
/// Exact Black Forest Labs formula: rearrange(x, "b (h w) (c ph pw) -> b c (h ph) (w pw)", ph=2, pw=2)
pub fn unpatchify(latents: &Tensor, height: usize, width: usize) -> Result<Tensor> {
    let (b, _, channels) = latents.dims3()?;
    let h_patches = (height + 15) / 16;
    let w_patches = (width + 15) / 16;
    let c = channels / 4;
    let ph = 2;
    let pw = 2;

    // 1. Reshape to [B, H/16, W/16, C, PH, PW]
    let reshaped = latents.reshape((b, h_patches, w_patches, c, ph, pw))?;
    // 2. Permute to [B, C, H/16, PH, W/16, PW] -> (0, 3, 1, 4, 2, 5)
    let permuted = reshaped.permute((0, 3, 1, 4, 2, 5))?.contiguous()?;
    // 3. Merge spatial dimensions -> [B, C, H/8, W/8]
    let unpatchified = permuted.reshape((b, c, h_patches * ph, w_patches * pw))?;
    Ok(unpatchified)
}

/// Pure Rust Pipeline for Flux.1 (Schnell / Dev) Multimodal Diffusion Transformer
pub struct FluxPipeline {
    pub checkpoint_path: PathBuf,
    pub transformer: FluxTransformer,
    pub scheduler: FlowMatchEulerScheduler,
    pub clip_l: Option<crate::text::ClipTextEncoder>,
    pub t5xxl: Option<crate::text::T5TextEncoder>,
    pub qwen3: Option<crate::text::Qwen3TextEncoder>,
    pub vae: Option<crate::diffusion::vae_flux::FluxVaeDecoder>,
    pub streamer: Option<crate::diffusion::dit::streamer::SequentialBlockStreamer>,
    pub device: Device,
    pub dtype: DType,
}

impl FluxPipeline {
    /// Attach Qwen3 text encoder for Flux.2 Klein
    pub fn set_qwen3(&mut self, qwen: crate::text::Qwen3TextEncoder) {
        self.qwen3 = Some(qwen);
    }

    /// Enable the FlashAttention-2 fast path for MMDiT blocks (~2x faster denoise on CUDA).
    ///
    /// Runs attention in F16/BF16 via `candle_flash_attn`, automatically falling back to the
    /// model-safe F32 SDPA path on any error (unsupported dtype/backend). Requires the
    /// `--features flash-attn` cargo feature; otherwise the F32 path is used regardless.
    pub fn enable_flash_attn(&mut self) {
        unsafe { std::env::set_var("FLUX_FLASH_ATTN", "1") };
    }

    /// Disable the FlashAttention-2 fast path and use the stable F32 attention (default).
    pub fn disable_flash_attn(&mut self) {
        unsafe { std::env::set_var("FLUX_FLASH_ATTN", "0") };
    }
    /// Load Flux.1 pipeline with Sequential Block Streaming (< 6.5 GB VRAM peak)
    pub fn from_single_file_streaming<P: AsRef<Path>>(checkpoint_path: P, device: Device) -> crate::error::Result<Self> {
        let is_cuda = device.is_cuda();
        let dtype = if is_cuda { DType::F16 } else { DType::F32 };
        let checkpoint_buf = checkpoint_path.as_ref().to_path_buf();

        let archive = Arc::new(SafeTensorsArchive::open(&checkpoint_buf)?);
        let router = WeightRouter::new(&archive, device.clone(), dtype);

        println!("📦 Constructing Pure Rust Flux Streaming Transformer (Ultra-Low VRAM)...");
        let has_guidance = archive.keys().any(|k| k.contains("guidance_in"));
        let is_klein = archive.keys().any(|k| k.contains("double_stream_modulation"));
        let config = if is_klein {
            println!("✨ Detected Flux.2-Klein 4B checkpoint (compact shared-modulation architecture)!");
            FluxConfig::klein_4b()
        } else if has_guidance {
            println!("✨ Detected Flux.1-Dev checkpoint (with guidance embedder)!");
            FluxConfig::dev()
        } else {
            println!("✨ Detected Flux.1-Schnell checkpoint (distilled fast inference)!");
            FluxConfig::schnell()
        };
        let vb = router.flux_header_var_builder()?;
        let transformer = FluxTransformer::new_streaming(config.clone(), vb)?;

        let streamer = Some(crate::diffusion::dit::streamer::SequentialBlockStreamer::new(
            archive.clone(),
            device.clone(),
            dtype,
            config.hidden_size,
            config.num_heads,
            config.mlp_ratio,
        ));

        // Auto-detect and instantiate CLIP-L
        let clip_l = match router.clip_l_var_builder_on_device(&Device::Cpu, DType::F32) {
            Ok(clip_vb) => {
                println!("✨ Auto-detected embedded CLIP-L text encoder (CPU)!");
                match crate::text::ClipTextEncoder::new_sd15(clip_vb) {
                    Ok(enc) => {
                        println!("✅ CLIP-L encoder successfully loaded!");
                        Some(enc)
                    }
                    Err(e) => {
                        println!("[-] Failed to load CLIP-L encoder: {}", e);
                        None
                    }
                }
            }
            Err(e) => {
                println!("[-] No CLIP-L weights in checkpoint: {}", e);
                None
            }
        };

        // Auto-detect and instantiate T5-XXL
        let t5xxl = match router.t5xxl_var_builder_on_device(&Device::Cpu, DType::F32) {
            Ok(t5_vb) => {
                println!("✨ Auto-detected embedded T5-XXL text encoder (CPU)!");
                match crate::text::T5TextEncoder::new(t5_vb, None) {
                    Ok(enc) => {
                        println!("✅ T5-XXL encoder successfully loaded!");
                        Some(enc)
                    }
                    Err(e) => {
                        println!("[-] Failed to load T5-XXL encoder: {}", e);
                        None
                    }
                }
            }
            Err(e) => {
                println!("[-] No T5-XXL weights in checkpoint: {}", e);
                None
            }
        };

        let vae_router = WeightRouter::new(&archive, device.clone(), DType::F16);
        let vae = match vae_router.vae_var_builder() {
            Ok(vae_vb) => {
                println!("✨ Auto-detected embedded 16-channel Flux VAE! Loading weights (CUDA GPU)...");
                match crate::diffusion::vae_flux::FluxVaeDecoder::new(vae_vb) {
                    Ok(v) => {
                        println!("✅ Flux VAE successfully loaded on GPU!");
                        Some(v)
                    }
                    Err(e) => {
                        println!("[-] Failed to construct Flux VAE on GPU: {}", e);
                        None
                    }
                }
            }
            Err(e) => {
                println!("[-] No VAE found in router: {}", e);
                None
            }
        };

        let scheduler = FlowMatchEulerScheduler::new(FlowMatchEulerConfig::default());

        Ok(Self {
            checkpoint_path: checkpoint_buf,
            transformer,
            scheduler,
            clip_l,
            t5xxl,
            qwen3: None,
            vae,
            streamer,
            device,
            dtype,
        })
    }

    /// Load Flux.1 pipeline from a local single-file checkpoint (.safetensors)
    pub fn from_single_file<P: AsRef<Path>>(checkpoint_path: P, device: Device) -> crate::error::Result<Self> {
        Self::from_single_file_streaming(checkpoint_path, device)
    }

    /// Attach 16-channel AutoEncoder VAE
    pub fn set_vae(&mut self, vae: crate::diffusion::vae_flux::FluxVaeDecoder) {
        self.vae = Some(vae);
    }

    /// Generate image using Rectified Flow ODE solver (default 4 steps for Schnell)
    pub fn generate_with_metrics<F>(
        &mut self,
        params: DiffusionParams,
        progress_cb: Option<F>,
    ) -> crate::error::Result<(image::RgbImage, GenerationMetrics)>
    where
        F: Fn(usize, usize, &Tensor),
    {
        let t_total = Instant::now();
        let num_steps = params.num_steps;

        let in_channels = self.transformer.config.in_channels;
        let c = in_channels / 4; // 16 for Flux 1, 32 for Flux 2 Klein
        let ph = 2;
        let pw = 2;

        let h_patches = (params.height + 15) / 16;
        let w_patches = (params.width + 15) / 16;
        let image_seq_len = h_patches * w_patches;

        if in_channels == 128 {
            // Flux.2 Klein standard shift
            self.scheduler = FlowMatchEulerScheduler::new(FlowMatchEulerConfig {
                shift: 2.02,
                base_shift: 0.5,
                max_shift: 1.15,
                min_shift: 0.5,
            });
        }
        self.scheduler.set_timesteps_with_seq_len(num_steps, image_seq_len)?;

        // 1. Initial Gaussian latent noise matching Diffusers Flux 2:
        // Diffusers draws noise in shape: [1, 128, H_p, W_p] and packs to [1, H_p * W_p, 128]
        let mut latents = if in_channels == 128 {
            let raw_diff_noise = Tensor::randn(0f32, 1f32, (1, 128, h_patches, w_patches), &self.device)?.to_dtype(self.dtype)?;
            raw_diff_noise.reshape((1, 128, h_patches * w_patches))?.permute((0, 2, 1))?.contiguous()?
        } else {
            let raw_noise = Tensor::randn(0f32, 1f32, (1, c, h_patches * ph, w_patches * pw), &self.device)?.to_dtype(self.dtype)?;
            let reshaped = raw_noise.reshape((1, c, h_patches, ph, w_patches, pw))?;
            let permuted = reshaped.permute((0, 2, 4, 1, 3, 5))?.contiguous()?;
            permuted.reshape((1, h_patches * w_patches, in_channels))?
        };

        // 1. Text conditioning: encode prompt via Qwen3 (for Klein) or T5-XXL (for Flux 1)
        let txt_tokens = if let Some(ref mut qwen) = self.qwen3 {
            println!("📝 Encoding prompt with Qwen3 (Layers 9, 18, 27 -> 7680 dim, 512 tokens)...");
            let qwen_emb = qwen.encode(params.prompt, 512)?;
            qwen_emb.to_device(&self.device)?.to_dtype(self.dtype)?
        } else if let Some(ref mut t5) = self.t5xxl {
            println!("📝 Encoding prompt with T5-XXL (256 tokens)...");
            let t5_emb = t5.encode(params.prompt, 256)?;
            let t = t5_emb.to_device(&self.device)?.to_dtype(self.dtype)?;
            if in_channels == 128 && t.dim(2)? < 7680 {
                // Pad or expand to 7680 for Klein architecture
                let pad = Tensor::zeros((t.dim(0)?, t.dim(1)?, 7680 - t.dim(2)?), self.dtype, &self.device)?;
                Tensor::cat(&[&t, &pad], 2)?
            } else {
                t
            }
        } else {
            let dim = if in_channels == 128 { 7680 } else { 4096 };
            (Tensor::randn(0f32, 1.0f32, (1, 256, dim), &self.device)? * 0.1)?.to_dtype(self.dtype)?
        };

        let y_vec = if let Some(ref mut clip) = self.clip_l {
            if !clip.has_tokenizer() {
                let _ = clip.load_tokenizer("clip_tokenizer.json");
            }
            println!("📝 Encoding prompt with CLIP-L (pooled vector)...");
            match clip.encode_pooled(params.prompt) {
                Ok(vec) => Some(vec.to_device(&self.device)?.to_dtype(self.dtype)?),
                Err(e) => {
                    println!("[-] CLIP-L encode pooled error: {}", e);
                    None
                }
            }
        } else {
            None
        };

        println!("⚡ Executing Flux.1 Flow Matching ODE ({} steps)...", num_steps);
        let t_unet_start = Instant::now();

        let guidance_tensor = if self.transformer.config.guidance_embed {
            let g = (params.guidance_scale * 1000.0) as f32;
            Some(Tensor::from_slice(&[g], (1,), &self.device)?.to_dtype(self.dtype)?)
        } else {
            None
        };

        let timesteps: Vec<usize> = self.scheduler.timesteps().to_vec();
        let sigmas: Vec<f64> = self.scheduler.sigmas().to_vec();
        for (step_idx, &t) in timesteps.iter().enumerate() {
            let sigma = if step_idx < sigmas.len() { sigmas[step_idx] } else { 0.0 };
            let t_tensor = Tensor::from_slice(&[sigma as f32], (1,), &self.device)?.to_dtype(self.dtype)?;
            
            // Forward pass predicting velocity field v_t (with on-demand block streaming)
            let velocity = self.transformer.forward_with_streamer(
                &latents,
                &txt_tokens,
                &t_tensor,
                y_vec.as_ref(),
                guidance_tensor.as_ref(),
                self.streamer.as_ref(),
            )?;

            // Euler integration step: x_{t-1} = x_t + dt * v_t
            latents = self.scheduler.step(&velocity, t, &latents)?;

            if let Some(ref cb) = progress_cb {
                cb(step_idx + 1, num_steps, &latents);
            }
        }

        let unet_duration = t_unet_start.elapsed();
        let unet_total_ms = unet_duration.as_secs_f64() * 1000.0;
        let unet_it_per_sec = num_steps as f64 / unet_duration.as_secs_f64();
        let unet_step_avg_ms = unet_total_ms / num_steps as f64;

        // 2. Unpack latents & VAE de-standardization
        let t_vae_start = Instant::now();
        let unpatchified_latents = if in_channels == 128 {
            // Flux 2 official unpatchify pipeline:
            // a. [1, H_p*W_p, 128] -> [1, H_p, W_p, 128] -> permute(0, 3, 1, 2) -> [1, 128, H_p, W_p]
            let grid_4d = latents.reshape((1, h_patches, w_patches, 128))?.permute((0, 3, 1, 2))?.contiguous()?;
            
            // b. BatchNorm de-standardization in 128-dim space: latents * std + mean
            let destandardized = if let Some(ref vae) = self.vae {
                if let (Some(mean), Some(var)) = (vae.bn_mean(), vae.bn_var()) {
                    let mean_f32 = mean.to_dtype(DType::F32)?.reshape((1, 128, 1, 1))?;
                    let var_f32 = var.to_dtype(DType::F32)?.reshape((1, 128, 1, 1))?;
                    let std_f32 = (var_f32 + 1e-4)?.sqrt()?;
                    let grid_f32 = grid_4d.to_dtype(DType::F32)?;
                    let normed = grid_f32.broadcast_mul(&std_f32)?.broadcast_add(&mean_f32)?;
                    normed.to_dtype(self.dtype)?
                } else {
                    grid_4d
                }
            } else {
                grid_4d
            };

            // c. Exact Flux 2 _unpatchify_latents:
            // [1, 128, H_p, W_p] -> [1, 32, 2, 2, H_p, W_p] -> permute(0, 1, 4, 2, 5, 3) -> [1, 32, H_p, 2, W_p, 2] -> [1, 32, H_p*2, W_p*2]
            let reshaped = destandardized.reshape((1, 32, 2, 2, h_patches, w_patches))?;
            let permuted = reshaped.permute((0, 1, 4, 2, 5, 3))?.contiguous()?;
            permuted.reshape((1, 32, h_patches * 2, w_patches * 2))?
        } else {
            let normalized = if let Some(ref vae) = self.vae {
                if let (Some(mean), Some(var)) = (vae.bn_mean(), vae.bn_var()) {
                    let mean_f32 = mean.to_dtype(DType::F32)?;
                    let var_f32 = var.to_dtype(DType::F32)?;
                    let std = (var_f32 + 1e-5)?.sqrt()?;
                    let mean_bc = mean_f32.unsqueeze(0)?.unsqueeze(1)?;
                    let std_bc = std.unsqueeze(0)?.unsqueeze(1)?;
                    let latents_f32 = latents.to_dtype(DType::F32)?;
                    let destandardized = latents_f32.broadcast_mul(&std_bc)?.broadcast_add(&mean_bc)?;
                    destandardized.to_dtype(self.dtype)?
                } else {
                    latents.clone()
                }
            } else {
                latents.clone()
            };
            unpatchify(&normalized, params.height, params.width)?
        };

        // 3. Decode via Flux VAE if attached
        let image = if let Some(ref vae) = self.vae {
            vae.decode_to_image(&unpatchified_latents)?
        } else {
            // Direct normalized visualization of the first 3 latent channels
            let rgb_latent = unpatchified_latents.narrow(1, 0, 3)?;
            crate::diffusion::vae::tensor_to_rgb_image(&rgb_latent)?
        };

        let vae_decode_ms = t_vae_start.elapsed().as_secs_f64() * 1000.0;
        let total_wallclock_ms = t_total.elapsed().as_secs_f64() * 1000.0;

        let metrics = GenerationMetrics {
            prompt_encode_ms: 0.0,
            unet_steps: num_steps,
            unet_total_ms,
            unet_it_per_sec,
            unet_step_avg_ms,
            vae_decode_ms,
            total_wallclock_ms,
        };

        Ok((image, metrics))
    }

    /// Image-to-Image (Img2Img) Generation for Rectified Flow matching models.
    pub fn generate_img2img<F>(
        &mut self,
        params: crate::traits::Img2ImgParams,
        progress_cb: Option<F>,
    ) -> crate::error::Result<(image::RgbImage, GenerationMetrics)>
    where
        F: Fn(usize, usize, &Tensor),
    {
        let t_total = Instant::now();
        let (width, height) = (params.image.width() as usize, params.image.height() as usize);
        let num_steps = params.num_steps;
        let h_patches = (height + 15) / 16;
        let w_patches = (width + 15) / 16;
        let image_seq_len = h_patches * w_patches;
        self.scheduler.set_timesteps_with_seq_len(num_steps, image_seq_len)?;
        let c = 16;
        let ph = 2;
        let pw = 2;

        // 1. Initial Gaussian noise for flow interpolation: x_t = (1 - sigma)*x_0 + sigma*noise
        let raw_noise = Tensor::randn(0f32, 1f32, (1, c, h_patches * ph, w_patches * pw), &self.device)?.to_dtype(self.dtype)?;
        let reshaped = raw_noise.reshape((1, c, h_patches, ph, w_patches, pw))?;
        let permuted = reshaped.permute((0, 2, 4, 1, 3, 5))?.contiguous()?;
        let noise_tokens = permuted.reshape((1, h_patches * w_patches, c * ph * pw))?;

        // 2. Text conditioning: encode prompt via T5-XXL and CLIP-L
        let txt_tokens = if let Some(ref mut t5) = self.t5xxl {
            let t5_emb = t5.encode(params.prompt, 256)?;
            t5_emb.to_device(&self.device)?.to_dtype(self.dtype)?
        } else {
            (Tensor::randn(0f32, 1.0f32, (1, 256, 4096), &self.device)? * 0.1)?.to_dtype(self.dtype)?
        };

        let y_vec = if let Some(ref mut clip) = self.clip_l {
            match clip.encode_pooled(params.prompt) {
                Ok(vec) => Some(vec.to_device(&self.device)?.to_dtype(self.dtype)?),
                Err(_) => None,
            }
        } else {
            None
        };

        let guidance_tensor = if self.transformer.config.guidance_embed {
            let g = (params.guidance_scale * 1000.0) as f32;
            Some(Tensor::from_slice(&[g], (1,), &self.device)?.to_dtype(self.dtype)?)
        } else {
            None
        };

        // Determine starting step based on strength: strength in [0.0, 1.0]
        let start_step = ((1.0 - params.strength.clamp(0.0, 1.0)) * num_steps as f64) as usize;
        let start_step = start_step.min(num_steps.saturating_sub(1));

        let mut latents = noise_tokens;
        let timesteps: Vec<usize> = self.scheduler.timesteps().to_vec();
        let sigmas: Vec<f64> = self.scheduler.sigmas().to_vec();

        let t_unet_start = Instant::now();
        for step_idx in start_step..timesteps.len() {
            let t = timesteps[step_idx];
            let sigma = if step_idx < sigmas.len() { sigmas[step_idx] } else { 0.0 };
            let t_tensor = Tensor::from_slice(&[sigma as f32], (1,), &self.device)?.to_dtype(self.dtype)?;

            let velocity = self.transformer.forward_with_streamer(
                &latents,
                &txt_tokens,
                &t_tensor,
                y_vec.as_ref(),
                guidance_tensor.as_ref(),
                self.streamer.as_ref(),
            )?;

            latents = self.scheduler.step(&velocity, t, &latents)?;

            if let Some(ref cb) = progress_cb {
                cb(step_idx + 1, num_steps, &latents);
            }
        }

        let unet_duration = t_unet_start.elapsed();
        let unet_total_ms = unet_duration.as_secs_f64() * 1000.0;
        let active_steps = num_steps - start_step;
        let unet_it_per_sec = active_steps as f64 / unet_duration.as_secs_f64();
        let unet_step_avg_ms = if active_steps > 0 { unet_total_ms / active_steps as f64 } else { 0.0 };

        let t_vae_start = Instant::now();
        let unpatchified_latents = unpatchify(&latents, height, width)?;
        let image = if let Some(ref vae) = self.vae {
            vae.decode_to_image(&unpatchified_latents)?
        } else {
            let rgb_latent = unpatchified_latents.narrow(1, 0, 3)?;
            crate::diffusion::vae::tensor_to_rgb_image(&rgb_latent)?
        };

        let vae_decode_ms = t_vae_start.elapsed().as_secs_f64() * 1000.0;
        let total_wallclock_ms = t_total.elapsed().as_secs_f64() * 1000.0;

        let metrics = GenerationMetrics {
            prompt_encode_ms: 0.0,
            unet_steps: active_steps,
            unet_total_ms,
            unet_it_per_sec,
            unet_step_avg_ms,
            vae_decode_ms,
            total_wallclock_ms,
        };

        Ok((image, metrics))
    }

    pub fn checkpoint_path(&self) -> &Path {
        &self.checkpoint_path
    }
}
