// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: ACE-Step 5Hz LM planner (Qwen3 causal LM) port

//! ACE-Step 1.5 "5Hz LM" planner: a fine-tuned Qwen3 causal LM that turns a
//! simple caption (+ lyrics) into CoT metadata / audio semantic tokens which
//! guide the diffusion DiT. Reuses the generic [`CausalLMPipeline`] engine.

use std::path::Path;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device};
use tokenizers::Tokenizer;

use crate::models::text::{CausalLMConfig, CausalLMPipeline, TextGenParams};
use crate::weights::{SafeTensorsArchive, WeightsSource};

/// ACE-Step 5Hz language-model planner.
pub struct AceStepLm {
    pub pipeline: CausalLMPipeline,
}

impl AceStepLm {
    /// Default planner instruction (`constants.DEFAULT_LM_INSTRUCTION`).
    pub const DEFAULT_LM_INSTRUCTION: &'static str =
        "Generate audio semantic tokens based on the given conditions:";
    pub const EOS_TOKEN_ID: u32 = 151645;
    pub const PAD_TOKEN_ID: u32 = 151643;

    /// Load the LM from a HuggingFace-style directory (`model.safetensors` + `tokenizer.json`).
    pub fn from_dir<P: AsRef<Path>>(dir: P, device: &Device, dtype: DType) -> Result<Self> {
        let dir = dir.as_ref();
        let archive: Arc<dyn WeightsSource> = Arc::new(
            SafeTensorsArchive::open_shards_dir(dir)
                .with_context(|| format!("failed to open LM weights in {:?}", dir))?,
        );
        let config = CausalLMConfig::from_weights(&*archive)?;
        let tok_path = dir.join("tokenizer.json");
        let tokenizer = if tok_path.exists() {
            Some(Tokenizer::from_file(&tok_path).map_err(|e| anyhow!("tokenizer load failed: {e}"))?)
        } else {
            None
        };
        let pipeline = CausalLMPipeline::new(archive, config, tokenizer, device.clone(), dtype);
        Ok(Self { pipeline })
    }

    /// Qwen chat-formatted prompt mirroring `LLMHandler.build_formatted_prompt`
    /// (system instruction + `# Caption` / `# Lyric` user block + generation prompt).
    pub fn build_formatted_prompt(caption: &str, lyrics: &str) -> String {
        let system = format!("# Instruction\n{}\n\n", Self::DEFAULT_LM_INSTRUCTION);
        let user = format!("# Caption\n{}\n\n# Lyric\n{}\n", caption, lyrics);
        format!(
            "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            system, user
        )
    }

    /// Raw text generation with explicit sampling parameters.
    pub fn generate(&mut self, prompt: &str, params: &TextGenParams) -> Result<String> {
        self.pipeline.generate(prompt, params).map_err(|e| anyhow!("{e}"))
    }

    /// Plan metadata / lyrics / CoT for a caption (the LM "thinking" phase).
    pub fn plan(&mut self, caption: &str, lyrics: &str, max_tokens: usize) -> Result<String> {
        let prompt = Self::build_formatted_prompt(caption, lyrics);
        let params = TextGenParams {
            max_tokens,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            repetition_penalty: 1.05,
            stop_tokens: vec![Self::EOS_TOKEN_ID, Self::PAD_TOKEN_ID],
        };
        self.generate(&prompt, &params)
    }
}
