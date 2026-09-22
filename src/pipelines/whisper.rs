// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: Pure Rust Speech-to-Text (STT) Whisper Pipeline

use std::path::Path;
use std::time::Instant;
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{audio::pcm_to_mel, model::Whisper, Config};
use tokenizers::Tokenizer;

use crate::audio::{whisper_mel_filters, WavAudio};

/// Speech transcription result.
#[derive(Debug, Clone, PartialEq)]
pub struct TranscriptionResult {
    /// Full transcribed text.
    pub text: String,
    /// Detected or requested language code (e.g. "en", "fr").
    pub language: String,
    /// Duration of input audio in seconds.
    pub duration_seconds: f32,
    /// Total wallclock inference duration in milliseconds.
    pub inference_time_ms: f64,
}

/// Pure Rust Whisper Speech-to-Text (STT) Pipeline.
pub struct WhisperPipeline {
    pub model: Whisper,
    pub tokenizer: Tokenizer,
    pub config: Config,
    pub mel_filters: Vec<f32>,
    pub device: Device,
    pub dtype: DType,
}

impl WhisperPipeline {
    /// Create a new Whisper pipeline.
    pub fn new(
        model: Whisper,
        tokenizer: Tokenizer,
        config: Config,
        device: Device,
        dtype: DType,
    ) -> Self {
        let mel_filters = whisper_mel_filters(config.num_mel_bins);
        Self {
            model,
            tokenizer,
            config,
            mel_filters,
            device,
            dtype,
        }
    }

    /// Load Whisper pipeline from local file paths.
    pub fn from_files(
        config_path: &Path,
        weights_path: &Path,
        tokenizer_path: &Path,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        let config_str = std::fs::read_to_string(config_path)
            .with_context(|| format!("Failed to read Whisper config at {:?}", config_path))?;
        let config: Config = serde_json::from_str(&config_str)
            .context("Failed to deserialize Whisper config")?;

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load Whisper tokenizer: {}", e))?;

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, &device)
                .with_context(|| format!("Failed to load Whisper weights from {:?}", weights_path))?
        };

        let model = Whisper::load(&vb, config.clone())
            .context("Failed to initialize Whisper model")?;

        Ok(Self::new(model, tokenizer, config, device, dtype))
    }

    /// Override the Mel filterbank coefficients.
    pub fn set_mel_filters(&mut self, filters: Vec<f32>) {
        self.mel_filters = filters;
    }

    /// Transcribe an audio buffer into text.
    ///
    /// # Arguments
    /// * `audio` - The input audio waveform.
    /// * `language` - Target language ("en", "fr", etc.). Defaults to English "en" if None.
    /// * `timestamps` - Whether to include timestamp tokens.
    pub fn transcribe(
        &mut self,
        audio: &WavAudio,
        language: Option<&str>,
        timestamps: bool,
    ) -> Result<TranscriptionResult> {
        let t_start = Instant::now();
        let lang_str = language.unwrap_or("en");

        // 1. Ensure audio is mono and 16,000 Hz for Whisper
        let processed_audio = audio.to_mono().resample(16000);
        let mono_samples = processed_audio.samples;

        // 2. Compute Log-Mel Spectrogram across the full audio sequence
        let mel = pcm_to_mel(&self.config, &mono_samples, &self.mel_filters);
        let mel_len = mel.len();
        let num_mel_bins = self.config.num_mel_bins;
        let mel_tensor = Tensor::from_vec(
            mel,
            (1, num_mel_bins, mel_len / num_mel_bins),
            &self.device,
        )?.to_dtype(self.dtype)?;

        // 3. Encode audio features
        let audio_features = self.model.encoder.forward(&mel_tensor, true)
            .context("Whisper encoder forward pass failed")?;

        // 4. Decode autoregressively
        let sot_token = 50258u32; // <|startoftranscript|>
        let eot_token = 50257u32; // <|endoftranscript|>
        let trans_token = 50359u32; // <|transcribe|>
        let notimestamps_token = 50363u32; // <|notimestamps|>

        // Find language token id from tokenizer
        let lang_token = self.tokenizer.token_to_id(&format!("<|{}|>", lang_str))
            .unwrap_or(50259); // Default to <|en|> (50259)

        let mut tokens = vec![sot_token, lang_token, trans_token];
        if !timestamps {
            tokens.push(notimestamps_token);
        }

        let suppress_tokens: Vec<f32> = (0..self.config.vocab_size as u32)
            .map(|i| {
                if self.config.suppress_tokens.contains(&i) || (timestamps && i == notimestamps_token) {
                    f32::NEG_INFINITY
                } else {
                    0.0f32
                }
            })
            .collect();
        let suppress_tokens_tensor = Tensor::new(suppress_tokens.as_slice(), &self.device)?
            .to_dtype(self.dtype)?;

        self.model.decoder.reset_kv_cache();
        let mut generated_tokens = Vec::new();
        let max_target_positions = self.config.max_target_positions.min(448);

        for i in 0..max_target_positions {
            let tokens_t = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;
            let ys = self.model.decoder.forward(&tokens_t, &audio_features, i == 0)?;

            let (_, seq_len, _) = ys.dims3()?;
            let last_hidden = ys.narrow(1, seq_len - 1, 1)?;
            let logits = self.model.decoder.final_linear(&last_hidden)?;
            let logits_1d = logits.squeeze(0)?.squeeze(0)?;
            let filtered_logits = logits_1d.broadcast_add(&suppress_tokens_tensor)?;

            // Greedy argmax
            let next_token = filtered_logits.argmax(0)?.to_scalar::<u32>()?;

            if next_token == eot_token {
                break;
            }

            tokens.push(next_token);
            generated_tokens.push(next_token);
        }

        let text = self.tokenizer.decode(&generated_tokens, true)
            .map_err(|e| anyhow::anyhow!("Token decoding failed: {}", e))?;

        let inference_time_ms = t_start.elapsed().as_secs_f64() * 1000.0;

        Ok(TranscriptionResult {
            text: text.trim().to_string(),
            language: lang_str.to_string(),
            duration_seconds: audio.duration_seconds(),
            inference_time_ms,
        })
    }
}
