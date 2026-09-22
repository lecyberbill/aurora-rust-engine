// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: Pure Rust Neural Text-to-Speech (TTS) Pipeline

use std::path::Path;
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::dac;
use candle_transformers::models::parler_tts;
use tokenizers::Tokenizer;

use crate::audio::WavAudio;

/// Neural Text-to-Speech synthesis pipeline.
///
/// Combines a prompt-conditioned autoregressive transformer (Parler-TTS)
/// with a neural acoustic codec (Descript Audio Codec - DAC) to generate
/// broadcast-quality speech waveforms from text and voice descriptions.
pub struct TtsPipeline {
    pub model: parler_tts::Model,
    pub dac: dac::Model,
    pub tokenizer: Tokenizer,
    pub description_tokenizer: Tokenizer,
    pub config: parler_tts::Config,
    pub device: Device,
}

impl TtsPipeline {
    /// Create a new TTS pipeline from loaded components.
    pub fn new(
        model: parler_tts::Model,
        dac: dac::Model,
        tokenizer: Tokenizer,
        description_tokenizer: Tokenizer,
        config: parler_tts::Config,
        device: Device,
    ) -> Self {
        Self {
            model,
            dac,
            tokenizer,
            description_tokenizer,
            config,
            device,
        }
    }

    /// Load pipeline from local file paths.
    pub fn from_files(
        config_path: &Path,
        weights_path: &Path,
        dac_weights_path: &Path,
        tokenizer_path: &Path,
        description_tokenizer_path: &Path,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        let config_str = std::fs::read_to_string(config_path)
            .with_context(|| format!("Failed to read config at {:?}", config_path))?;
        let config: parler_tts::Config = serde_json::from_str(&config_str)
            .context("Failed to deserialize Parler-TTS config")?;

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
        let description_tokenizer = Tokenizer::from_file(description_tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load description tokenizer: {}", e))?;

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, &device)
                .with_context(|| format!("Failed to load weights from {:?}", weights_path))?
        };

        let dac_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[dac_weights_path], dtype, &device)
                .with_context(|| format!("Failed to load DAC weights from {:?}", dac_weights_path))?
        };

        let model = parler_tts::Model::new(&config, vb)
            .context("Failed to initialize Parler-TTS model")?;
        let dac_model = dac::Model::new(&config.audio_encoder, dac_vb)
            .context("Failed to initialize DAC model")?;

        Ok(Self::new(
            model,
            dac_model,
            tokenizer,
            description_tokenizer,
            config,
            device,
        ))
    }

    /// Synthesize speech from text and a natural voice description.
    ///
    /// # Arguments
    /// * `prompt` - The text to be spoken.
    /// * `description` - The natural voice description (e.g. "A female speaker delivers a slightly expressive speech with a moderate pitch.")
    /// * `max_steps` - Maximum autoregressive token generation steps (default: 500-1000).
    /// * `temperature` - Sampling temperature (default: 0.7 - 1.0).
    /// * `seed` - Random seed for generation reproducibility.
    pub fn synthesize(
        &mut self,
        prompt: &str,
        description: &str,
        max_steps: usize,
        temperature: f64,
        seed: u64,
    ) -> Result<WavAudio> {
        let prompt_encoding = self.tokenizer.encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed for prompt: {}", e))?;
        let desc_encoding = self.description_tokenizer.encode(description, true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed for description: {}", e))?;

        let prompt_ids: Vec<u32> = prompt_encoding.get_ids().to_vec();
        let desc_ids: Vec<u32> = desc_encoding.get_ids().to_vec();

        let prompt_tokens = Tensor::new(prompt_ids.as_slice(), &self.device)?
            .unsqueeze(0)?;
        let desc_tokens = Tensor::new(desc_ids.as_slice(), &self.device)?
            .unsqueeze(0)?;

        let lp = LogitsProcessor::new(seed, Some(temperature), None);

        // Generate discrete acoustic codebook tokens
        let audio_codes = self.model.generate(
            &prompt_tokens,
            &desc_tokens,
            lp,
            max_steps,
        ).context("Failed during Parler-TTS autoregressive token generation")?;

        // Decode discrete codes into continuous waveform using DAC
        let waveform = self.dac.decode_codes(&audio_codes)
            .context("Failed to decode acoustic codes with DAC vocoder")?;

        // Squeeze batch & channel dimensions: [1, 1, samples] -> [samples]
        let flat_waveform = waveform.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
        let samples: Vec<f32> = flat_waveform.to_vec1()?;

        let sample_rate = self.config.audio_encoder.sampling_rate;

        Ok(WavAudio::new(samples, sample_rate, 1))
    }
}
