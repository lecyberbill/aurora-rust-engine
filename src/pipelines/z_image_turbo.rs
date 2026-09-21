// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: Pure Rust Z-Image Turbo 1-4 Step Inference Pipeline

use candle_core::{DType, Device, Result, Tensor};
use image::RgbImage;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use crate::device::GenerationMetrics;
use crate::diffusion::dit::z_image::{ZImageConfig, ZImageTransformer};
use crate::diffusion::schedulers::{FlowMatchEulerConfig, FlowMatchEulerScheduler, Scheduler};
use crate::diffusion::vae_flux::FluxVaeDecoder;
use crate::text::Qwen3TextEncoder;
use crate::traits::DiffusionParams;
use crate::weights::SafeTensorsArchive;

/// Pipeline for Z-Image Turbo Realtime DiT Model
pub struct ZImageTurboPipeline {
    pub transformer: ZImageTransformer,
    pub scheduler: FlowMatchEulerScheduler,
    pub text_encoder: Option<Qwen3TextEncoder>,
    pub vae: Option<FluxVaeDecoder>,
    pub device: Device,
    pub dtype: DType,
}

impl ZImageTurboPipeline {
    /// Load from an all-in-one checkpoint (e.g. `z-image-turbo-fp8-aio.safetensors`)
    pub fn from_aio_checkpoint(
        path: impl AsRef<Path>,
        vae_path: Option<impl AsRef<Path>>,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let p = path.as_ref();
        let archive = Arc::new(
            SafeTensorsArchive::open(p)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?
        );

        // 1. Build DiT Transformer
        println!("🚀 Loading Z-Image Turbo DiT Transformer from {:?}", p);
        let mut dit_tensors = std::collections::HashMap::new();
        for key in archive.keys() {
            if let Some(rest) = key.strip_prefix("model.diffusion_model.") {
                if let Ok(t) = archive.get_tensor(&key, device, dtype) {
                    dit_tensors.insert(rest.to_string(), t);
                }
            }
        }
        let dit_vb = candle_nn::VarBuilder::from_tensors(dit_tensors, dtype, device);
        let config = ZImageConfig::default();
        let transformer = ZImageTransformer::new(config, dit_vb)?;

        // 2. Build Qwen3 Text Encoder from same AIO checkpoint
        println!("🧠 Loading Qwen3 Text Encoder from AIO checkpoint");
        let text_encoder = match Qwen3TextEncoder::from_archive(archive.as_ref(), None, device, dtype) {
            Ok(te) => {
                println!("✅ Qwen3 Text Encoder successfully instantiated (has tokenizer: {})", te.has_tokenizer());
                Some(te)
            }
            Err(e) => {
                println!("⚠️ Qwen3 Text Encoder failed to load: {:?}", e);
                None
            }
        };

        // 3. Flow Match Euler Scheduler (Official Z-Image Turbo dynamic shift: 0.5 to 1.15)
        let scheduler_cfg = FlowMatchEulerConfig {
            shift: 1.0,
            base_shift: 0.5,
            max_shift: 1.15,
            min_shift: 0.5,
            use_dynamic_shifting: true,
            double_shift_linspace: false,
        };
        let scheduler = FlowMatchEulerScheduler::new(scheduler_cfg);

        // 4. Load VAE Decoder on CPU to prevent GPU VRAM spike during full DiT inference
        let vae_device = Device::Cpu;
        let vae = if let Some(vp) = vae_path {
            let vae_archive = Arc::new(
                SafeTensorsArchive::open(vp.as_ref())
                    .map_err(|e| candle_core::Error::Msg(e.to_string()))?
            );
            let mut vae_tensors = std::collections::HashMap::new();
            for key in vae_archive.keys() {
                if let Ok(t) = vae_archive.get_tensor(&key, &vae_device, DType::F32) {
                    vae_tensors.insert(key.to_string(), t);
                }
            }
            let vae_vb = candle_nn::VarBuilder::from_tensors(vae_tensors, DType::F32, &vae_device);
            FluxVaeDecoder::new(vae_vb).ok()
        } else {
            // Check if VAE is embedded in the AIO archive
            let mut vae_tensors = std::collections::HashMap::new();
            for key in archive.keys() {
                if let Some(rest) = key.strip_prefix("vae.") {
                    if let Ok(t) = archive.get_tensor(&key, &vae_device, DType::F32) {
                        vae_tensors.insert(rest.to_string(), t);
                    }
                }
            }
            if !vae_tensors.is_empty() {
                let vae_vb = candle_nn::VarBuilder::from_tensors(vae_tensors, DType::F32, &vae_device);
                FluxVaeDecoder::new(vae_vb).ok()
            } else {
                None
            }
        };

        Ok(Self {
            transformer,
            scheduler,
            text_encoder,
            vae,
            device: device.clone(),
            dtype,
        })
    }

    /// Attach or override VAE Decoder
    pub fn set_vae(&mut self, vae: FluxVaeDecoder) {
        self.vae = Some(vae);
    }

    /// Enable FlashAttention-2 fast path (CUDA F16/BF16)
    pub fn enable_flash_attn(&mut self) {
        unsafe { std::env::set_var("ZIMAGE_FLASH_ATTN", "1") };
    }

    /// Disable FlashAttention-2 and use standard SDPA
    pub fn disable_flash_attn(&mut self) {
        unsafe { std::env::set_var("ZIMAGE_FLASH_ATTN", "0") };
    }

    /// Generate image from prompt
    pub fn generate(&mut self, params: &DiffusionParams) -> Result<(RgbImage, GenerationMetrics)> {
        let total_start = Instant::now();
        let height = params.height;
        let width = params.width;
        let num_steps = params.num_steps.clamp(1, 8); // Z-Image Turbo is optimized for 1-4 steps
        let b = 1;

        println!(
            "⚡ Z-Image Turbo Inference: prompt='{}' | steps={} | size={}x{}",
            params.prompt, num_steps, width, height
        );

        // 1. Text Encoding (Qwen3 -> context [1, seq_len, 2560])
        let t_encode_start = Instant::now();
        let max_seq_len = 0; // 0 = dynamic exact prompt length (no noise padding)
        let context = if let Some(ref enc) = self.text_encoder {
            enc.encode_last_hidden(params.prompt, max_seq_len)?
        } else {
            Tensor::zeros((1, 1, 2560), self.dtype, &self.device)?
        };
        let text_encoding_time_ms = t_encode_start.elapsed().as_millis() as f64;

        // 2. Initialise Latent Tensor: [1, 16, H/8, W/8]
        let latent_h = height / 8;
        let latent_w = width / 8;
        let mut latents = Tensor::randn(0.0f32, 1.0f32, (b, 16, latent_h, latent_w), &self.device)?
            .to_dtype(self.dtype)?;

        // 3. Scheduler Timesteps Setup with image sequence length
        let image_seq_len = (latent_h / 2) * (latent_w / 2);
        self.scheduler.set_timesteps_with_seq_len(num_steps, image_seq_len)?;
        let timesteps = self.scheduler.timesteps().to_vec();

        // 4. Denoising Loop
        let t_denoise_start = Instant::now();
        for (step_idx, &t) in timesteps.iter().enumerate() {
            let step_start = Instant::now();
            let sigma = self.scheduler.sigmas()[step_idx] as f32;
            let lumina_t = (1.0f32 - sigma) * 1000.0f32;
            let t_tensor = Tensor::from_vec(vec![lumina_t], (1,), &self.device)?.to_dtype(self.dtype)?;

            // Model Forward: predicts velocity
            let pred_v = self.transformer.forward(&latents, &t_tensor, &context)?;
            let v_std = pred_v.to_dtype(DType::F32)?.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt();

            // Negate velocity for Flow Matching Euler ODE step:
            // Official diffusers pipeline_z_image: noise_pred = -noise_pred; scheduler.step(noise_pred, ...)
            let noise_pred = pred_v.neg()?;

            // Step scheduler
            latents = self.scheduler.step(&noise_pred, t, &latents)?;
            let lat_std = latents.to_dtype(DType::F32)?.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt();

            println!(
                "   [Step {}/{}] sigma={:.4} | pred_v std={:.4} | latents std={:.4} ({:.1} ms)",
                step_idx + 1,
                num_steps,
                sigma,
                v_std,
                lat_std,
                step_start.elapsed().as_secs_f64() * 1000.0
            );

            // Decode intermediate step (only if explicitly enabled for diagnostics)
            if std::env::var("ZIMAGE_DEBUG_STEPS").ok().map(|s| s == "1").unwrap_or(false) {
                if let Some(ref vae) = self.vae {
                    if let Ok(cpu_lat) = latents.to_device(&Device::Cpu)?.to_dtype(DType::F32) {
                        if let Ok(dec) = vae.decode(&cpu_lat) {
                            if let Ok(step_img) = crate::diffusion::vae::tensor_to_rgb_image(&dec) {
                                let _ = step_img.save(format!("output/step_{}.png", step_idx + 1));
                            }
                        }
                    }
                }
            }
        }
        let total_denoising_time_ms = t_denoise_start.elapsed().as_millis() as f64;

        // 5. Decode Latents through VAE (on CPU to prevent VRAM spikes)
        let t_vae_start = Instant::now();
        let vae = self.vae.as_ref().ok_or_else(|| {
            candle_core::Error::Msg("VAE Decoder missing for Z-Image Turbo pipeline".into())
        })?;

        let cpu_latents = latents.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
        let decoded = vae.decode(&cpu_latents)?;
        let vae_decode_time_ms = t_vae_start.elapsed().as_millis() as f64;

        // 6. Convert decoded tensor to RGB Image
        let rgb_img = crate::diffusion::vae::tensor_to_rgb_image(&decoded)?;
        let total_time_ms = total_start.elapsed().as_millis() as f64;

        let metrics = GenerationMetrics {
            prompt_encode_ms: text_encoding_time_ms,
            unet_steps: num_steps,
            unet_total_ms: total_denoising_time_ms,
            unet_step_avg_ms: total_denoising_time_ms / num_steps as f64,
            unet_it_per_sec: if total_denoising_time_ms > 0.0 { (num_steps as f64 * 1000.0) / total_denoising_time_ms } else { 0.0 },
            vae_decode_ms: vae_decode_time_ms,
            total_wallclock_ms: total_time_ms,
        };

        Ok((rgb_img, metrics))
    }
}
