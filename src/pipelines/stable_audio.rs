// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: Stable Audio Open text-to-audio pipeline (pure Rust)

//! Stable Audio Open pipeline: T5 prompt + seconds conditioning → EDM/SDE DPM-Solver++ DiT →
//! Oobleck 44.1 kHz VAE decode. Ported from `diffusers.StableAudioPipeline`.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use std::path::Path;

use crate::audio::{seeded_randn, AutoencoderOobleck, OobleckConfig, WavAudio};
use crate::models::{
    CosineDpmScheduler, StableAudioDit, StableAudioProjectionModel, T5Encoder,
};

pub struct StableAudioPipeline {
    pub t5: T5Encoder,
    pub projection: StableAudioProjectionModel,
    pub dit: StableAudioDit,
    pub vae: AutoencoderOobleck,
    pub tokenizer: tokenizers::Tokenizer,
    pub device: Device,
    pub dtype: DType,
    pub sample_size: usize,
    pub sample_rate: u32,
}

impl StableAudioPipeline {
    pub fn from_pretrained<P: AsRef<Path>>(model_dir: P) -> Result<Self> {
        let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
        let dtype = if device.is_cuda() { DType::F32 } else { DType::F32 };
        Self::from_folder(model_dir, device, dtype)
    }

    pub fn from_folder<P: AsRef<Path>>(dir: P, device: Device, dtype: DType) -> Result<Self> {
        let dir = dir.as_ref();
        let t5 = T5Encoder::from_safetensors(dir.join("text_encoder/model.safetensors"), &device, dtype)
            .context("T5")?;
        let projection = StableAudioProjectionModel::from_safetensors(
            dir.join("projection_model/diffusion_pytorch_model.safetensors"),
            &device,
            dtype,
        )
        .context("projection")?;
        let dit = StableAudioDit::from_safetensors(
            dir.join("transformer/diffusion_pytorch_model.safetensors"),
            &device,
            dtype,
        )
        .context("DiT")?;
        let vae_cfg = OobleckConfig {
            audio_channels: 2,
            channel_multiples: vec![1, 2, 4, 8, 16],
            decoder_channels: 128,
            decoder_input_channels: 64,
            downsampling_ratios: vec![2, 4, 4, 8, 8],
            sampling_rate: 44100,
        };
        let vae = AutoencoderOobleck::from_safetensors(
            dir.join("vae/diffusion_pytorch_model.safetensors"),
            vae_cfg,
            &device,
            dtype,
        )
        .context("VAE")?;
        let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer/tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
        Ok(Self {
            t5,
            projection,
            dit,
            vae,
            tokenizer,
            device,
            dtype,
            sample_size: 1024,
            sample_rate: 44100,
        })
    }

    fn encode_text(&self, text: &str) -> Result<(Tensor, Tensor)> {
        let enc = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
        let mut ids: Vec<u32> = enc.get_ids().to_vec();
        let mut mask: Vec<f32> = vec![1.0; ids.len()];
        let max_len = 128usize;
        let real = ids.len().min(max_len);
        ids.truncate(max_len);
        mask.truncate(max_len);
        while ids.len() < max_len {
            ids.push(0);
            mask.push(0.0);
        }
        let _ = real;
        let b = 1usize;
        let ids_t = Tensor::from_vec(ids, (b, max_len), &self.device)?;
        let am = Tensor::from_vec(mask, (b, max_len), &self.device)?.to_dtype(self.dtype)?;
        let hidden = self.t5.forward(&ids_t, &am.to_dtype(DType::U32)?)?;
        // projection text (identity) then mask
        let proj = self
            .projection
            .forward(Some(&hidden), None, None)?
            .text_hidden_states
            .unwrap();
        let m = am.unsqueeze(2)?;
        Ok((proj.broadcast_mul(&m)?, am))
    }

    fn number_cond(&self, start: f32, end: f32) -> Result<(Tensor, Tensor)> {
        let s = Tensor::new(&[start], &self.device)?.to_dtype(self.dtype)?;
        let e = Tensor::new(&[end], &self.device)?.to_dtype(self.dtype)?;
        let out = self.projection.forward(None, Some(&s), Some(&e))?;
        Ok((
            out.seconds_start_hidden_states.unwrap(),
            out.seconds_end_hidden_states.unwrap(),
        ))
    }

    /// Generate 44.1 kHz stereo audio. Returns `(WavAudio, duration_seconds)`.
    #[allow(clippy::too_many_arguments)]
    pub fn generate(
        &self,
        prompt: &str,
        negative: &str,
        start_seconds: f32,
        end_seconds: f32,
        steps: usize,
        guidance_scale: f32,
        seed: u64,
    ) -> Result<WavAudio> {
        let (pos_text, _am) = self.encode_text(prompt)?; // [1,128,768]
        let (s, e) = self.number_cond(start_seconds, end_seconds)?; // [1,1,768]

        let (prompt_embeds, b) = if negative.trim().is_empty() {
            (pos_text, 1usize)
        } else {
            let (neg_text, _) = self.encode_text(negative)?;
            (Tensor::cat(&[&neg_text, &pos_text], 0)?, 2usize)
        };
        let s2 = s.repeat((b, 1, 1))?;
        let e2 = e.repeat((b, 1, 1))?;
        let mut cond = Tensor::cat(&[&prompt_embeds, &s2, &e2], 1)?; // [b,130,768]
        let mut global = Tensor::cat(&[&s2, &e2], 2)?; // [b,1,1536]
        if b == 1 {
            // No negative prompt: unconditional branch = zeros for text, same duration embeds.
            let zeros = Tensor::zeros_like(&cond)?;
            cond = Tensor::cat(&[&zeros, &cond], 0)?;
            global = Tensor::cat(&[&global, &global], 0)?;
        }

        let mut sched = CosineDpmScheduler::new(steps);
        let init_sigma = sched.init_noise_sigma() as f32;
        let mut latents = seeded_randn(self.sample_size, 64, 0.0, 1.0, seed, self.dtype, &self.device)?
            .transpose(1, 2)?
            .contiguous()?
            .affine(init_sigma as f64, 0.0)?; // [1,64,1024]
        if std::env::var("SAO_DEBUG").is_ok() {
            let e = self.tokenizer.encode(prompt, true).map_err(|x| anyhow::anyhow!("{x}"))?;
            let ids = e.get_ids();
            eprintln!("dbg ids_len={} head={:?}", ids.len(), &ids[..20.min(ids.len())]);
            eprintln!(
                "dbg cond|mean|={:.4} global|mean|={:.4} lat0 std={:.3}",
                cond.abs()?.mean_all()?.to_scalar::<f32>()?,
                global.abs()?.mean_all()?.to_scalar::<f32>()?,
                latents.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?,
            );
        }

        let dbg = std::env::var("SAO_DEBUG").is_ok();
        for i in 0..steps {
            let inp = Tensor::cat(&[&latents, &latents], 0)?; // [2,64,1024]
            let inp = sched.scale_input(&inp, i)?;
            let t = Tensor::new(&[sched.timesteps[i] as f32], &self.device)?.to_dtype(self.dtype)?;
            let pred = self.dit.forward(&inp, &t, &cond, &global)?; // [2,64,1024]
            let parts = pred.chunk(2, 0)?;
            let guided = (&parts[0] + &(&parts[1] - &parts[0])?.affine(guidance_scale as f64, 0.0)?)?;
            // SDE noise: independent N(0,1) per step (Brownian increments over disjoint intervals).
            let noise = seeded_randn(self.sample_size, 64, 0.0, 1.0, seed.wrapping_add(i as u64 + 1), self.dtype, &self.device)?
                .transpose(1, 2)?
                .contiguous()?;
            latents = sched.step(&guided, &latents, i, &noise)?;
            if dbg && i % 20 == 0 {
                eprintln!(
                    "step {i}: pred|mean|={:.4} latents std={:.3}",
                    pred.abs()?.mean_all()?.to_scalar::<f32>()?,
                    latents.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?
                );
            }
        }

        let audio = self.vae.decode(&latents)?;
        // Crop to the requested window (fields are interleaved stereo).
        let ch = self.vae.config.audio_channels;
        let end_frame = (end_seconds * self.sample_rate as f32) as usize;
        let want = end_frame * ch;
        if audio.samples.len() > want {
            let mut s = audio.samples;
            s.truncate(want);
            return Ok(WavAudio::new(s, self.sample_rate, ch as u16));
        }
        Ok(audio)
    }
}
