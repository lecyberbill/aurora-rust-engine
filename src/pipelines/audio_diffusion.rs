// [WFGY] Zone: SAFE | λ: 0.25 | Fallbacks: 0 | Action: Pure Rust Text-to-Audio & Music Diffusion Pipeline (ACE-Step 1.5 Turbo + Oobleck 48kHz VAE)

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use std::path::Path;
use std::time::Instant;

use crate::audio::{seeded_randn, AutoencoderOobleck, AudioFormat, OobleckConfig, WavAudio};
use crate::models::acestep_tasks;
use crate::models::{AceStepConditionEncoder, AceStepTransformer1D, AceStepTransformerConfig, FlowMatchConfig};
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

/// Source-audio task request (cover / repaint / extract / lego / complete).
///
/// Drives the reference conditioning path: task instruction, caption (incl. the
/// SFT-stems `Global:/Local:/Mask Control:` block), VAE-encoded source latents,
/// optional repaint span, and Flow-Matching sampler settings.
#[derive(Debug, Clone)]
pub struct TaskRequest<'a> {
    /// Task name (`AceStepTask` string form, e.g. `"extract"`, `"lego"`).
    pub task: &'a str,
    /// Explicit instruction; empty derives it from `task`/`track`/`classes`.
    pub instruction: &'a str,
    pub caption: &'a str,
    /// Full-song description (SFT-stems `Global:` prefix), only used when the
    /// checkpoint sets `is_lego_sft`.
    pub global_caption: &'a str,
    pub lyrics: &'a str,
    pub language: &'a str,
    pub track_name: Option<&'a str>,
    pub complete_track_classes: Option<Vec<String>>,
    /// `[1, T, 64]` frame-major VAE-encoded source latents.
    pub src_latents: &'a Tensor,
    /// Optional `[1, T, 64]` reference-audio latents for the timbre encoder
    /// (`None` → learned silence latent, matching the reference default).
    pub refer_latents: Option<&'a Tensor>,
    /// Repaint span in seconds (repaint/lego only; `None` → full-length).
    pub repaint_start: Option<f32>,
    pub repaint_end: Option<f32>,
    pub steps: usize,
    pub guidance_scale: f32,
    pub seed: u64,
    /// Chunk-mask fill value for non-repaint tasks (1.0 = plain, 2.0 = "auto"/Mask Control).
    pub chunk_mask_value: f32,
    /// Fraction of steps conditioned on the cover source (1.0 = always).
    pub cover_strength: f32,
    /// `> 0` initializes `x_t` from the source at the nearest timestep.
    pub cover_noise_strength: f32,
}

impl<'a> TaskRequest<'a> {
    pub fn new(task: &'a str, src_latents: &'a Tensor) -> Self {
        Self {
            task,
            instruction: "",
            caption: "",
            global_caption: "",
            lyrics: "[instrumental]",
            language: "en",
            track_name: None,
            complete_track_classes: None,
            src_latents,
            refer_latents: None,
            repaint_start: None,
            repaint_end: None,
            steps: 8,
            guidance_scale: 1.0,
            seed: 42,
            // "auto"/Mask Control value used by extract/lego/complete (repaint uses explicit 0/1).
            chunk_mask_value: 2.0,
            cover_strength: 1.0,
            cover_noise_strength: 0.0,
        }
    }

    pub fn with_instruction(mut self, instruction: &'a str) -> Self {
        self.instruction = instruction;
        self
    }
    pub fn with_caption(mut self, caption: &'a str) -> Self {
        self.caption = caption;
        self
    }
    pub fn with_global_caption(mut self, global_caption: &'a str) -> Self {
        self.global_caption = global_caption;
        self
    }
    pub fn with_lyrics(mut self, lyrics: &'a str) -> Self {
        self.lyrics = lyrics;
        self
    }
    pub fn with_language(mut self, language: &'a str) -> Self {
        self.language = language;
        self
    }
    pub fn with_track_name(mut self, track_name: &'a str) -> Self {
        self.track_name = Some(track_name);
        self
    }
    pub fn with_complete_track_classes(mut self, classes: Vec<String>) -> Self {
        self.complete_track_classes = Some(classes);
        self
    }
    pub fn with_repaint_span(mut self, start_seconds: f32, end_seconds: f32) -> Self {
        self.repaint_start = Some(start_seconds);
        self.repaint_end = Some(end_seconds);
        self
    }
    pub fn with_steps(mut self, steps: usize) -> Self {
        self.steps = steps;
        self
    }
    pub fn with_guidance_scale(mut self, guidance_scale: f32) -> Self {
        self.guidance_scale = guidance_scale;
        self
    }
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
    pub fn with_cover_strength(mut self, cover_strength: f32) -> Self {
        self.cover_strength = cover_strength;
        self
    }
    pub fn with_cover_noise_strength(mut self, cover_noise_strength: f32) -> Self {
        self.cover_noise_strength = cover_noise_strength;
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
    /// Set on SFT-stems checkpoints (`model.config.is_lego_sft`); enables the
    /// `Global:/Local:/Mask Control:` caption block for lego/extract/complete.
    pub is_lego_sft: bool,
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
            is_lego_sft: false,
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
                pipeline.is_lego_sft = v.get("is_lego_sft").and_then(|b| b.as_bool()).unwrap_or(false);
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
        self.build_conditioning_ex(
            acestep_tasks::DEFAULT_DIT_INSTRUCTION,
            caption,
            lyrics,
            language,
            duration_seconds,
            num_frames,
            None,
            None,
        )
    }

    /// Like [`Self::build_conditioning`] but with a task `instruction`, explicit reference-audio
    /// latents for the timbre encoder, and an explicit `src_latents` (cover / repaint / LM hints).
    #[allow(clippy::too_many_arguments)]
    pub fn build_conditioning_ex(
        &self,
        instruction: &str,
        caption: &str,
        lyrics: &str,
        language: &str,
        duration_seconds: f32,
        num_frames: usize,
        refer_latents: Option<&Tensor>,
        src_latents: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let (condition, src_64, chunk_64) = if let (Some(te), Some(ce)) =
            (&self.text_encoder, &self.condition_encoder)
        {
            let metas = acestep_tasks::default_metas(duration_seconds);
            let text_prompt = acestep_tasks::format_sft_caption(instruction, caption, &metas);
            let lyrics_text = acestep_tasks::format_lyrics(lyrics, language);

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
                .forward_condition_ex(&text_hidden, &lyric_embeds, refer_latents)
                .map_err(|e| anyhow::anyhow!("Condition encoding failed: {}", e))?;

            let src = match src_latents {
                Some(s) => s.to_dtype(self.dtype)?,
                None => {
                    let sil = ce.silence_latent.to_dtype(self.dtype)?;
                    let avail = sil.dim(1)?;
                    if num_frames <= avail {
                        sil.narrow(1, 0, num_frames)?
                    } else {
                        let reps = (num_frames + avail - 1) / avail;
                        sil.repeat((1, reps, 1))?.narrow(1, 0, num_frames)?
                    }
                }
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

    /// Tile the learned silence latent `[1, avail, 64]` to `num_frames` frames.
    fn silence_context(&self, num_frames: usize) -> Result<Tensor> {
        let ce = self
            .condition_encoder
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("condition encoder required for task conditioning"))?;
        let sil = ce.silence_latent.to_dtype(self.dtype)?;
        let avail = sil.dim(1)?;
        let tiled = if num_frames <= avail {
            sil.narrow(1, 0, num_frames)?
        } else {
            let reps = (num_frames + avail - 1) / avail;
            sil.repeat((1, reps, 1))?.narrow(1, 0, num_frames)?
        };
        Ok(tiled)
    }

    /// Run a source-audio task (cover / repaint / extract / lego / complete),
    /// mirroring `ConditioningMaskMixin._build_chunk_masks_and_src_latents` plus the
    /// reference `AceStepConditionGenerationModel` sampler.
    pub fn generate_task(&self, req: &TaskRequest) -> Result<(WavAudio, AudioGenerationMetrics)> {
        let t0 = Instant::now();
        let task = acestep_tasks::AceStepTask::parse(req.task);
        let num_frames = req.src_latents.dim(1)?;
        let duration = num_frames as f32 / 25.0;

        let instruction = if req.instruction.is_empty() {
            acestep_tasks::generate_instruction(
                task.as_str(),
                req.track_name,
                req.complete_track_classes.as_deref(),
            )
        } else {
            acestep_tasks::format_instruction(req.instruction)
        };
        let caption = if self.is_lego_sft {
            acestep_tasks::lego_sft_caption(req.caption, req.global_caption, &instruction, "true")
        } else {
            req.caption.to_string()
        };

        let (condition, _) = self.build_conditioning_ex(
            &instruction,
            &caption,
            req.lyrics,
            req.language,
            duration,
            num_frames,
            req.refer_latents,
            None,
        )?;

        // Repaint span in latent frames (`can_use_repainting` = repaint/lego only).
        let span = if task.uses_repaint() {
            match (req.repaint_start, req.repaint_end) {
                (Some(s), Some(e)) if e > s => {
                    let sl = ((s.max(0.0) * 25.0) as usize).min(num_frames.saturating_sub(1));
                    let el = ((e * 25.0) as usize).clamp(sl + 1, num_frames);
                    Some((sl, el))
                }
                _ => None,
            }
        } else {
            None
        };

        let src = req.src_latents.to_dtype(self.dtype)?;
        let mut src_task_vec: Vec<f32> = src.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        // `chunk_mask`: explicit 0/1 for repaint; all-ones (boolean `True` in the
        // reference, the "auto" mode 2.0 is a no-op on a bool tensor) otherwise.
        let mut chunk_vec = vec![req.chunk_mask_value; num_frames];
        let repaint_mask = if let Some((sl, el)) = span {
            let mut rm = vec![0.0f32; num_frames];
            for k in sl..el {
                rm[k] = 1.0;
            }
            if task == acestep_tasks::AceStepTask::Repaint {
                for v in chunk_vec.iter_mut() {
                    *v = 0.0;
                }
                for k in sl..el {
                    chunk_vec[k] = 1.0;
                }
            }
            // Repaint silences the masked region; lego keeps the source as context.
            if !task.is_lego() {
                let sil_vec: Vec<f32> = self
                    .silence_context(num_frames)?
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1()?;
                for k in (sl * 64)..(el * 64) {
                    src_task_vec[k] = sil_vec[k];
                }
            }
            Some(Tensor::from_vec(rm, (1, num_frames), &self.device)?.to_dtype(self.dtype)?)
        } else {
            None
        };

        let src_task = Tensor::from_vec(src_task_vec, (1, num_frames, 64), &self.device)?
            .to_dtype(self.dtype)?;
        let chunk = Tensor::from_vec(chunk_vec, (1, num_frames, 1), &self.device)?
            .broadcast_as((1, num_frames, 64))?
            .to_dtype(self.dtype)?;
        let context_latents = Tensor::cat(&[&src_task, &chunk], 2)?;

        let noise = seeded_randn(num_frames, 64, 0.0, 1.0, req.seed, self.dtype, &self.device)?;
        let t_schedule: Vec<f64> = (0..req.steps)
            .map(|i| 1.0 - (i as f64) / (req.steps as f64))
            .collect();
        let guidance = if req.guidance_scale > 0.0 {
            req.guidance_scale
        } else {
            self.default_guidance_scale
        };
        let null = if self.variant == AceStepVariant::Base && guidance > 1.0 {
            self.condition_encoder
                .as_ref()
                .map(|ce| {
                    ce.null_condition_emb
                        .to_dtype(self.dtype)
                        .and_then(|n| n.broadcast_as(condition.dims()))
                })
                .transpose()?
        } else {
            None
        };

        let cfg = FlowMatchConfig {
            condition: &condition,
            null_condition: null.as_ref(),
            condition_non_cover: None,
            null_condition_non_cover: None,
            context_latents: &context_latents,
            context_latents_non_cover: None,
            noise: &noise,
            t_schedule: &t_schedule,
            guidance_scale: guidance,
            cover_strength: req.cover_strength,
            cover_noise_strength: req.cover_noise_strength,
            clean_src: Some(&src),
            repaint_mask: repaint_mask.as_ref(),
            repaint_injection_ratio: 0.5,
            repaint_crossfade_frames: 10,
        };
        let latents = self.transformer.flow_match(&cfg)?;

        if std::env::var("ACESTEP_DUMP_TASK").is_ok() {
            use safetensors::tensor::{serialize_to_file, TensorView};
            let mut owned: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
            for (n, t) in [
                ("condition", &condition),
                ("context_latents", &context_latents),
                ("noise", &noise),
                ("src_latents", &src),
                ("final_latents", &latents),
            ] {
                let shape = t.dims().to_vec();
                let v: Vec<f32> = t.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
                let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                owned.push((n.to_string(), shape, bytes));
            }
            let views: Vec<(&str, TensorView)> = owned
                .iter()
                .map(|(n, s, b)| {
                    (
                        n.as_str(),
                        TensorView::new(safetensors::Dtype::F32, s.clone(), b.as_slice()).unwrap(),
                    )
                })
                .collect();
            std::fs::create_dir_all("outputs/audio_ref")?;
            serialize_to_file(
                views,
                &None,
                std::path::Path::new("outputs/audio_ref/task_rust.safetensors"),
            )?;
            println!("dumped outputs/audio_ref/task_rust.safetensors");
        }

        let audio = self.decode(&latents)?;

        let metrics = AudioGenerationMetrics {
            duration_seconds: audio.duration_seconds(),
            num_steps: req.steps,
            inference_time_ms: t0.elapsed().as_secs_f64() * 1000.0,
            sample_rate: audio.sample_rate,
            channels: audio.channels,
        };
        Ok((audio, metrics))
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
