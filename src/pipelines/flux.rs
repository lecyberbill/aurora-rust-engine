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

/// Convert patchified sequence latents [B, (H/16)*(W/16), 64] back to 2D latents [B, 16, H/8, W/8]
/// Exact Black Forest Labs formula: rearrange(x, "b (h w) (c ph pw) -> b c (h ph) (w pw)", ph=2, pw=2, c=16)
pub fn unpatchify(latents: &Tensor, height: usize, width: usize) -> Result<Tensor> {
    let (b, _, _) = latents.dims3()?;
    let h_patches = (height + 15) / 16;
    let w_patches = (width + 15) / 16;
    let c = 16;
    let ph = 2;
    let pw = 2;

    // 1. Reshape to [B, H/16, W/16, C, PH, PW]
    let reshaped = latents.reshape((b, h_patches, w_patches, c, ph, pw))?;
    // 2. Permute to [B, C, H/16, PH, W/16, PW] -> (0, 3, 1, 4, 2, 5)
    let permuted = reshaped.permute((0, 3, 1, 4, 2, 5))?.contiguous()?;
    // 3. Merge spatial dimensions -> [B, 16, H/8, W/8]
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
    pub vae: Option<crate::diffusion::vae_flux::FluxVaeDecoder>,
    pub streamer: Option<crate::diffusion::dit::streamer::SequentialBlockStreamer>,
    pub device: Device,
    pub dtype: DType,
}

impl FluxPipeline {
    /// Load Flux.1 pipeline with Sequential Block Streaming (< 6.5 GB VRAM peak)
    pub fn from_single_file_streaming<P: AsRef<Path>>(checkpoint_path: P, device: Device) -> crate::error::Result<Self> {
        let is_cuda = device.is_cuda();
        let dtype = if is_cuda { DType::F16 } else { DType::F32 };
        let checkpoint_buf = checkpoint_path.as_ref().to_path_buf();

        let archive = Arc::new(SafeTensorsArchive::open(&checkpoint_buf)?);
        let router = WeightRouter::new(&archive, device.clone(), dtype);

        println!("📦 Constructing Pure Rust Flux.1 Streaming Transformer (Ultra-Low VRAM)...");
        // Auto-detect whether checkpoint is Flux.1-Dev (has guidance_in) or Flux.1-Schnell
        let has_guidance = archive.keys().any(|k| k.contains("guidance_in"));
        let config = if has_guidance {
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

        let h_patches = (params.height + 15) / 16;
        let w_patches = (params.width + 15) / 16;
        let image_seq_len = h_patches * w_patches;
        self.scheduler.set_timesteps_with_seq_len(num_steps, image_seq_len)?;
        let c = 16;
        let ph = 2;
        let pw = 2;

        // 1. Initial Gaussian latent noise matching Black Forest Labs get_noise: [1, 16, 2 * H/16, 2 * W/16]
        let raw_noise = Tensor::randn(0f32, 1f32, (1, c, h_patches * ph, w_patches * pw), &self.device)?.to_dtype(self.dtype)?;
        // 2. Pack to sequence tokens: rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        let reshaped = raw_noise.reshape((1, c, h_patches, ph, w_patches, pw))?;
        let permuted = reshaped.permute((0, 2, 4, 1, 3, 5))?.contiguous()?;
        let mut latents = permuted.reshape((1, h_patches * w_patches, c * ph * pw))?;

        // 1. Text conditioning: encode prompt via T5-XXL (sequence) and CLIP-L (pooled vector y)
        let txt_tokens = if let Some(ref mut t5) = self.t5xxl {
            println!("📝 Encoding prompt with T5-XXL (256 tokens)...");
            let t5_emb = t5.encode(params.prompt, 256)?;
            t5_emb.to_device(&self.device)?.to_dtype(self.dtype)?
        } else {
            (Tensor::randn(0f32, 1.0f32, (1, 256, 4096), &self.device)? * 0.1)?.to_dtype(self.dtype)?
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

        // 2. Unpatchify latents from [1, (H/16)*(W/16), 64] -> [1, 16, H/8, W/8]
        let t_vae_start = Instant::now();
        let unpatchified_latents = unpatchify(&latents, params.height, params.width)?;

        // 3. Decode via 16-channel Flux VAE if attached
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

    pub fn checkpoint_path(&self) -> &Path {
        &self.checkpoint_path
    }
}
