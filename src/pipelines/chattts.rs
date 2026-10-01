// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: Pure Rust ChatTTS Orchestrator Pipeline

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use std::path::Path;
use tokenizers::Tokenizer;

use crate::audio::WavAudio;
use crate::models::chattts::{ChatTtsConfig, ChatTtsModel};

/// Generation parameters for conversational ChatTTS synthesis.
#[derive(Debug, Clone)]
pub struct ChatTtsParams {
    pub prompt: String,
    pub speaker_seed: Option<u64>,
    pub temperature: f64,
    pub max_steps: usize,
    pub seed: u64,
}

impl Default for ChatTtsParams {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            speaker_seed: Some(42),
            temperature: 0.7,
            max_steps: 1024,
            seed: 42,
        }
    }
}

/// Standalone ChatTTS Pipeline orchestrating tokenization, GPT prosody generation, DVAE decoding and Vocos vocoding.
pub struct ChatTtsPipeline {
    pub model: ChatTtsModel,
    pub tokenizer: Tokenizer,
    pub device: Device,
}

impl ChatTtsPipeline {
    pub fn new(model: ChatTtsModel, tokenizer: Tokenizer, device: Device) -> Self {
        Self {
            model,
            tokenizer,
            device,
        }
    }

    /// Load the ChatTTS pipeline from local files.
    pub fn from_files(
        gpt_path: &Path,
        dvae_path: &Path,
        vocos_path: &Path,
        tokenizer_path: &Path,
        config: Option<ChatTtsConfig>,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        let config = config.unwrap_or_default();
        let model = ChatTtsModel::load_from_safetensors(
            gpt_path,
            dvae_path,
            vocos_path,
            config,
            device.clone(),
            dtype,
        ).context("Failed to load ChatTTS model components")?;

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load ChatTTS tokenizer at {:?}: {}", tokenizer_path, e))?;

        Ok(Self::new(model, tokenizer, device))
    }

    /// Deterministically sample a speaker embedding from a random seed.
    pub fn sample_speaker_embedding(&self, seed: u64) -> Result<Tensor> {
        let spk_dim = self.model.config.spk_emb_dim;
        let mut rng = crate::audio::rng::SeededRng::new(seed);
        let mut spk_vec = Vec::with_capacity(spk_dim);
        for _ in 0..spk_dim {
            let val = rng.next_normal() * 0.5;
            spk_vec.push(val);
        }
        let spk_tensor = Tensor::from_vec(spk_vec, (1, spk_dim), &self.device)?;
        Ok(spk_tensor)
    }

    /// Synthesize conversational speech waveform from text with prosodic markers.
    pub fn synthesize(&mut self, params: ChatTtsParams) -> Result<WavAudio> {
        let encoding = self.tokenizer.encode(params.prompt.as_str(), true)
            .map_err(|e| anyhow::anyhow!("ChatTTS Tokenization error: {}", e))?;
        
        let token_ids: Vec<u32> = encoding.get_ids().to_vec();
        anyhow::ensure!(!token_ids.is_empty(), "Tokenized prompt is empty");

        let spk_seed = params.speaker_seed.unwrap_or(params.seed);
        let spk_emb = self.sample_speaker_embedding(spk_seed)?;

        self.model.synthesize(
            &token_ids,
            &spk_emb,
            params.max_steps,
            params.temperature,
            params.seed,
        )
    }
}
