// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: Pure Rust Speech-to-Text (STT) Whisper Pipeline

use std::path::Path;
use std::time::Instant;
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{self, audio::pcm_to_mel, model::Whisper, Config};
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

        // 1. Convert to mono PCM 16kHz f32
        let mono_samples: Vec<f32> = if audio.channels == 1 {
            audio.samples.clone()
        } else {
            // Average stereo channels to mono
            let n_frames = audio.samples.len() / (audio.channels as usize);
            let mut mono = Vec::with_capacity(n_frames);
            for i in 0..n_frames {
                let mut sum = 0.0f32;
                for c in 0..(audio.channels as usize) {
                    sum += audio.samples[i * (audio.channels as usize) + c];
                }
                mono.push(sum / (audio.channels as f32));
            }
            mono
        };

        // 2. Pad or truncate to 30s chunk (480,000 samples @ 16kHz)
        let chunk_samples = whisper::N_SAMPLES;
        let mut padded = mono_samples.clone();
        if padded.len() < chunk_samples {
            padded.resize(chunk_samples, 0.0);
        } else if padded.len() > chunk_samples {
            padded.truncate(chunk_samples);
        }

        // 3. Compute Log-Mel Spectrogram in pure Rust
        let mel = pcm_to_mel(&self.config, &padded, &self.mel_filters);
        let n_mel = self.config.num_mel_bins;
        let n_frames = whisper::N_FRAMES; // 3000

        let mel_tensor = Tensor::from_vec(mel, (1, n_mel, n_frames), &self.device)?
            .to_dtype(self.dtype)?;

        // 4. Encode audio features
        self.model.encoder.forward(&mel_tensor, true)?;
        let audio_features = self.model.encoder.forward(&mel_tensor, false)
            .context("Whisper encoder forward pass failed")?;

        // 5. Decode autoregressively
        let sot_token = 50258u32; // <|startoftranscript|>
        let eot_token = 50257u32; // <|endoftranscript|>
        let trans_token = 50359u32; // <|transcribe|>
        let notimestamps_token = 50363u32; // <|notimestamps|>

        // Find language token id from tokenizer
        let lang_token = self.tokenizer.token_to_id(&format!("<|{}|>", lang_str))
            .unwrap_or(50259); // Default to <|en|> (50259)

        let mut prompt_tokens = vec![sot_token, lang_token, trans_token];
        if !timestamps {
            prompt_tokens.push(notimestamps_token);
        }

        self.model.decoder.reset_kv_cache();
        let mut generated_tokens = Vec::new();
        let max_target_positions = self.config.max_target_positions.min(448);

        // Feed initial prompt tokens
        let mut curr_token_tensor = Tensor::new(prompt_tokens.as_slice(), &self.device)?
            .unsqueeze(0)?;

        let mut is_first = true;
        for _ in 0..max_target_positions {
            let logits = self.model.decoder.forward(&curr_token_tensor, &audio_features, is_first)?;
            is_first = false;

            let (_, seq_len, _) = logits.dims3()?;
            let last_logit = logits.narrow(1, seq_len - 1, 1)?.squeeze(1)?;
            let linear_logits = self.model.decoder.final_linear(&last_logit)?;

            // Greedy argmax selection
            let next_token = linear_logits.squeeze(0)?.argmax(0)?.to_scalar::<u32>()?;

            if next_token == eot_token {
                break;
            }

            generated_tokens.push(next_token);
            curr_token_tensor = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
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
