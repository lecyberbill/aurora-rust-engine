// [WFGY] Zone: SAFE | λ: 0.25 | Fallbacks: 0 | Action: Pure Rust Text-to-Audio & Music Diffusion Pipeline (ACE-Step 1.5 Turbo + Oobleck 48kHz VAE)

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use std::path::Path;
use std::time::Instant;

use crate::audio::{seeded_randn, AutoencoderOobleck, AudioFormat, OobleckConfig, WavAudio};
use crate::models::{AceStepConditionEncoder, AceStepTransformer1D, AceStepTransformerConfig};
use crate::text::Qwen3TextEncoder;

/// Telemetry metrics for Audio Diffusion synthesis
#[derive(Debug, Clone)]
pub struct AudioGenerationMetrics {
    pub duration_seconds: f32,
    pub num_steps: usize,
    pub inference_time_ms: f64,
    pub sample_rate: u32,
    pub channels: u16,
}

/// ACE-Step model family: Turbo (CFG-distilled, few steps) or Base/SFT (guided, many steps).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AceStepVariant {
    Turbo,
    Base,
}

impl Default for AceStepVariant {
    fn default() -> Self {
        Self::Turbo
    }
}

/// High-level text-to-music request.
#[derive(Debug, Clone)]
pub struct TextToMusicRequest<'a> {
    pub caption: &'a str,
    pub lyrics: &'a str,
    pub language: &'a str,
    pub duration_seconds: f32,
    pub steps: usize,
    pub seed: u64,
    /// Classifier-free guidance strength (ignored by Turbo, which is CFG-distilled).
    pub guidance_scale: f32,
}

impl<'a> TextToMusicRequest<'a> {
    pub fn new(caption: &'a str, lyrics: &'a str) -> Self {
        Self {
            caption,
            lyrics,
            language: "en",
            duration_seconds: 30.0,
            steps: 8,
            seed: 42,
            guidance_scale: 1.0,
        }
    }

    pub fn with_language(mut self, language: &'a str) -> Self {
        self.language = language;
        self
    }
    pub fn with_duration(mut self, seconds: f32) -> Self {
        self.duration_seconds = seconds;
        self
    }
    pub fn with_steps(mut self, steps: usize) -> Self {
        self.steps = steps;
        self
    }
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
    pub fn with_guidance_scale(mut self, guidance_scale: f32) -> Self {
        self.guidance_scale = guidance_scale;
        self
    }
}

/// End-to-End Pure Rust Generative Audio & Music Diffusion Pipeline
pub struct AudioDiffusionPipeline {
    pub transformer: AceStepTransformer1D,
    pub vae: AutoencoderOobleck,
    pub condition_encoder: Option<AceStepConditionEncoder>,
    pub text_encoder: Option<Qwen3TextEncoder>,
    pub device: Device,
    pub dtype: DType,
    pub variant: AceStepVariant,
    pub default_guidance_scale: f32,
}

impl AudioDiffusionPipeline {
    pub fn new(
        transformer: AceStepTransformer1D,
        vae: AutoencoderOobleck,
        condition_encoder: Option<AceStepConditionEncoder>,
        text_encoder: Option<Qwen3TextEncoder>,
        device: Device,
        dtype: DType,
    ) -> Self {
        Self {
            transformer,
            vae,
            condition_encoder,
            text_encoder,
            device,
            dtype,
            variant: AceStepVariant::Turbo,
            default_guidance_scale: 1.0,
        }
    }

    /// Convenience loader: picks CUDA (bf16) when available, otherwise CPU (f32).
    pub fn from_pretrained<P: AsRef<Path>>(model_dir: P) -> Result<Self> {
        let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
        let dtype = if device.is_cuda() { DType::BF16 } else { DType::F32 };
        Self::from_folder(model_dir, device, dtype)
    }

    /// Load the pipeline from local model paths
    pub fn from_folder<P: AsRef<Path>>(
        model_dir: P,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        let dir = model_dir.as_ref();
        let vae_path = dir.join("vae").join("diffusion_pytorch_model.safetensors");
        let trans_dir = dir.join("transformer");
        let trans_shard1 = trans_dir.join("diffusion_pytorch_model-00001-of-00002.safetensors");
        let trans_shard2 = trans_dir.join("diffusion_pytorch_model-00002-of-00002.safetensors");
        let trans_single = trans_dir.join("diffusion_pytorch_model.safetensors");
        // Support both the sharded and single-file diffusers layouts.
        let trans_shards: Vec<std::path::PathBuf> = if trans_shard1.exists() && trans_shard2.exists() {
            vec![trans_shard1, trans_shard2]
        } else {
            vec![trans_single]
        };
        let cond_path = dir.join("condition_encoder").join("diffusion_pytorch_model.safetensors");
        let te_path = dir.join("text_encoder").join("model.safetensors");
        let tok_path = dir.join("tokenizer").join("tokenizer.json");

        let vae_config = OobleckConfig::default();
        let vae = AutoencoderOobleck::from_safetensors(&vae_path, vae_config, &device, dtype)
            .context("Failed to load AutoencoderOobleck VAE")?;

        let trans_cfg = AceStepTransformerConfig::from_json_file(dir.join("transformer").join("config.json"))
            .unwrap_or_default();
        let transformer = AceStepTransformer1D::from_safetensors_shards_with_config(
            &trans_shards,
            &device,
            dtype,
            &trans_cfg,
        ).context("Failed to load AceStep 1D Transformer")?;

        let condition_encoder = if cond_path.exists() {
            AceStepConditionEncoder::from_safetensors(&cond_path, &device, dtype).ok()
        } else {
            None
        };

        let text_encoder = if te_path.exists() {
            Qwen3TextEncoder::from_safetensors(&te_path, Some(&tok_path), &device, dtype).ok()
        } else {
            None
        };

        let mut pipeline = Self::new(transformer, vae, condition_encoder, text_encoder, device, dtype);

        // Detect Turbo vs Base/SFT from the transformer config (`is_turbo`).
        let cfg_path = dir.join("transformer").join("config.json");
        if let Ok(txt) = std::fs::read_to_string(&cfg_path) {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&txt) {
                let is_turbo = v.get("is_turbo").and_then(|b| b.as_bool()).unwrap_or(true);
                if !is_turbo {
                    pipeline.variant = AceStepVariant::Base;
                    pipeline.default_guidance_scale = 7.0;
                }
            }
        }
        Ok(pipeline)
    }

    /// Number of latent frames for a target duration (48 kHz / 1920 = 25 fps, minimum 128).
    pub fn latent_frames(duration_seconds: f32) -> usize {
        (((duration_seconds * 48000.0) as usize) / 1920).max(128)
    }

    /// Build the `[1, L, 2048]` cross-attention conditioning and the `[1, T, 128]`
    /// context latents (`[src_latents, chunk_mask]`) for text2music.
    pub fn build_conditioning(
        &self,
        caption: &str,
        lyrics: &str,
        language: &str,
        duration_seconds: f32,
        num_frames: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (condition, src_64, chunk_64) = if let (Some(te), Some(ce)) =
            (&self.text_encoder, &self.condition_encoder)
        {
            let instruction = "Fill the audio semantic mask based on the given conditions:";
            let metas = format!(
                "- bpm: N/A\n- timesignature: N/A\n- keyscale: N/A\n- duration: {} seconds\n",
                duration_seconds as i32
            );
            let text_prompt = format!(
                "# Instruction\n{}\n\n# Caption\n{}\n\n# Metas\n{}<|endoftext|>\n",
                instruction, caption, metas
            );
            let lyrics_text =
                format!("# Languages\n{}\n\n# Lyric\n{}<|endoftext|>", language, lyrics);

            let (text_ids, _) = te
                .tokenize_raw(&text_prompt, 256)
                .map_err(|e| anyhow::anyhow!("Text tokenization failed: {}", e))?;
            let text_hidden = te
                .forward_last_hidden(&text_ids)
                .map_err(|e| anyhow::anyhow!("Text encoding failed: {}", e))?;
            let (lyric_ids, _) = te
                .tokenize_raw(&lyrics_text, 2048)
                .map_err(|e| anyhow::anyhow!("Lyric tokenization failed: {}", e))?;
            let lyric_embeds = te
                .embed_ids(&lyric_ids)
                .map_err(|e| anyhow::anyhow!("Lyric embedding failed: {}", e))?;
            let cond = ce
                .forward_condition(&text_hidden, &lyric_embeds)
                .map_err(|e| anyhow::anyhow!("Condition encoding failed: {}", e))?;

            let sil = ce.silence_latent.to_dtype(self.dtype)?;
            let avail = sil.dim(1)?;
            let src = if num_frames <= avail {
                sil.narrow(1, 0, num_frames)?
            } else {
                let reps = (num_frames + avail - 1) / avail;
                sil.repeat((1, reps, 1))?.narrow(1, 0, num_frames)?
            };
            let chunk = Tensor::ones((1, num_frames, 64), self.dtype, &self.device)?;
            (cond, src, chunk)
        } else {
            let cond = Tensor::zeros((1, 16, 2048), self.dtype, &self.device)?;
            let zero = Tensor::zeros((1, num_frames, 64), self.dtype, &self.device)?;
            (cond, zero.clone(), zero)
        };

        // context_latents = [src_latents, chunk_mask] -> [1, T, 128]
        let context_latents = Tensor::cat(&[&src_64, &chunk_64], 2)?;
        Ok((condition, context_latents))
    }

    /// Build `context_latents [1, T, 128] = [src_latents, chunk_mask]` with an explicit
    /// source (e.g. 5Hz codec hints for cover / LM-code conditioning). `src` is `[1, T, 64]`.
    pub fn context_from_src(&self, src: &Tensor, num_frames: usize) -> Result<Tensor> {
        let src = src.to_dtype(self.dtype)?;
        let chunk = Tensor::ones((1, num_frames, 64), self.dtype, &self.device)?;
        Ok(Tensor::cat(&[&src, &chunk], 2)?)
    }

    /// Run the Flow-Matching Euler sampler from explicit conditioning/context/noise.
    /// Returns the final latents `[1, 64, T]`.
    pub fn diffuse(
        &self,
        condition: &Tensor,
        context_latents: &Tensor,
        noise: &Tensor,
        num_steps: usize,
    ) -> Result<Tensor> {
        self.diffuse_guided(condition, context_latents, noise, num_steps, 1.0)
    }

    /// Guidance-aware sampler. `guidance_scale > 1.0` on a Base/SFT model runs
    /// classifier-free guidance (`null_condition_emb` + APG); Turbo ignores it
    /// (CFG is distilled into the weights).
    pub fn diffuse_guided(
        &self,
        condition: &Tensor,
        context_latents: &Tensor,
        noise: &Tensor,
        num_steps: usize,
        guidance_scale: f32,
    ) -> Result<Tensor> {
        let t_schedule: Vec<f64> = (0..num_steps)
            .map(|i| 1.0 - (i as f64) / (num_steps as f64))
            .collect();

        if self.variant == AceStepVariant::Base && guidance_scale > 1.0 {
            if let Some(ce) = &self.condition_encoder {
                let null = ce
                    .null_condition_emb
                    .to_dtype(self.dtype)?
                    .broadcast_as(condition.dims())?;
                return Ok(self.transformer.flow_match_euler_cfg(
                    condition,
                    &null,
                    context_latents,
                    noise,
                    &t_schedule,
                    guidance_scale,
                )?);
            }
        }
        Ok(self
            .transformer
            .flow_match_euler(condition, context_latents, noise, &t_schedule)?)
    }

    /// Decode latents `[1, 64, T]` into a 48 kHz stereo master (tiled VAE decode).
    pub fn decode(&self, latents: &Tensor) -> Result<WavAudio> {
        Ok(self.vae.decode_tiled(latents, 256, 32)?)
    }

    /// Synthesize high-fidelity master stereo audio using Flow-Matching 1D Diffusion.
    ///
    /// # Arguments
    /// * `caption` - Style / genre / instrument description of the piece
    /// * `lyrics` - Lyric text (`"[Instrumental]"` or empty for instrumental)
    /// * `language` - Vocal language code (e.g. `"en"`, `"fr"`)
    /// * `duration_seconds` - Desired duration in seconds (e.g. 5.0, 30.0)
    /// * `num_steps` - Number of Flow-Matching inference steps (8 for Turbo, ~50 for Base)
    /// * `seed` - Deterministic random seed for the initial latent noise
    pub fn generate(
        &self,
        caption: &str,
        lyrics: &str,
        language: &str,
        duration_seconds: f32,
        num_steps: usize,
        seed: u64,
    ) -> Result<(WavAudio, AudioGenerationMetrics)> {
        let req = TextToMusicRequest::new(caption, lyrics)
            .with_language(language)
            .with_duration(duration_seconds)
            .with_steps(num_steps)
            .with_seed(seed)
            .with_guidance_scale(self.default_guidance_scale);
        self.text_to_music(&req)
    }

    /// Primary library entry point: `caption + lyrics -> 48 kHz stereo master`.
    /// Fully deterministic in `req.seed`.
    pub fn text_to_music(
        &self,
        req: &TextToMusicRequest,
    ) -> Result<(WavAudio, AudioGenerationMetrics)> {
        let num_frames = Self::latent_frames(req.duration_seconds);
        let noise = seeded_randn(num_frames, 64, 0.0, 1.0, req.seed, self.dtype, &self.device)?;
        self.generate_with_noise_guided(
            req.caption,
            req.lyrics,
            req.language,
            req.duration_seconds,
            req.steps,
            &noise,
            req.guidance_scale,
        )
    }

    /// Generate and persist to `path` (codec from `format`, or the path extension).
    pub fn text_to_music_to_file<P: AsRef<Path>>(
        &self,
        req: &TextToMusicRequest,
        path: P,
        format: Option<AudioFormat>,
    ) -> Result<AudioGenerationMetrics> {
        let (audio, metrics) = self.text_to_music(req)?;
        match format {
            Some(f) => audio.save_encoded(&path, f)?,
            None => audio.save_auto(&path)?,
        }
        Ok(metrics)
    }

    /// Like [`Self::generate`] but with an explicit initial noise tensor `[1, T, 64]`.
    /// Enables bit-exact comparison against an external reference (e.g. PyTorch).
    pub fn generate_with_noise(
        &self,
        caption: &str,
        lyrics: &str,
        language: &str,
        duration_seconds: f32,
        num_steps: usize,
        noise: &Tensor,
    ) -> Result<(WavAudio, AudioGenerationMetrics)> {
        self.generate_with_noise_guided(
            caption,
            lyrics,
            language,
            duration_seconds,
            num_steps,
            noise,
            self.default_guidance_scale,
        )
    }

    /// Full-control variant of [`Self::generate_with_noise`] with an explicit guidance scale.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_with_noise_guided(
        &self,
        caption: &str,
        lyrics: &str,
        language: &str,
        duration_seconds: f32,
        num_steps: usize,
        noise: &Tensor,
        guidance_scale: f32,
    ) -> Result<(WavAudio, AudioGenerationMetrics)> {
        let t_start = Instant::now();
        let noise = noise.to_dtype(self.dtype)?;
        let num_frames = noise.dim(1)?;

        let (condition, context_latents) =
            self.build_conditioning(caption, lyrics, language, duration_seconds, num_frames)?;
        let latents = self.diffuse_guided(&condition, &context_latents, &noise, num_steps, guidance_scale)?;
        let audio = self.decode(&latents)?;

        let metrics = AudioGenerationMetrics {
            duration_seconds: audio.duration_seconds(),
            num_steps,
            inference_time_ms: t_start.elapsed().as_secs_f64() * 1000.0,
            sample_rate: audio.sample_rate,
            channels: audio.channels,
        };
        Ok((audio, metrics))
    }
}
