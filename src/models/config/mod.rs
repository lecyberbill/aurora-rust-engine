// [WFGY] Zone: SAFE | λ: 0.25 | Fallbacks: 0 | Action: Centralised architecture sniffing for AutoModel (HF `AutoConfig`-like)

use crate::error::{LuminaError, Result};
use crate::weights::WeightsSource;
use super::common::ModelKind;

/// The resolved architecture family of a checkpoint, independent of the concrete pipeline.
///
/// This is the single place that maps *what a checkpoint looks like* to *what model family it is*,
/// mirroring HF's `AutoConfig.from_pretrained` where `model_type` + weight shapes decide. The
/// diffusion pipelines (`FluxPipeline`, SDXL/15) still keep their own per-constructor sniffing for
/// backward compatibility; this module is the **shared** table used by `AutoModel`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Architecture {
    Flux2Dev,
    Flux2Klein4B,
    Flux2Klein9B,
    Flux1Dev,
    Flux1Schnell,
    Sd35Large,
    Sd35Medium,
    Qwen3,
    Llama,
    DeepSeek,
    Gemma,
    Mistral3,
    T5,
    ClipL,
    OpenClip,
    Sdxl,
    Sd15,
    Unknown(String),
}

impl Architecture {
    pub fn model_kind(&self) -> ModelKind {
        match self {
            Architecture::Llama | Architecture::DeepSeek | Architecture::Gemma => ModelKind::Text,
            Architecture::Qwen3 | Architecture::Mistral3 => ModelKind::Text,
            Architecture::T5 | Architecture::ClipL | Architecture::OpenClip => ModelKind::TextEncoder,
            Architecture::Flux2Dev
            | Architecture::Flux2Klein4B
            | Architecture::Flux2Klein9B
            | Architecture::Flux1Dev
            | Architecture::Flux1Schnell
            | Architecture::Sd35Large
            | Architecture::Sd35Medium
            | Architecture::Sdxl
            | Architecture::Sd15 => ModelKind::Diffusion,
            Architecture::Unknown(_) => ModelKind::Unknown,
        }
    }

    pub fn slug(&self) -> String {
        match self {
            Architecture::Flux2Dev => "flux2-dev".into(),
            Architecture::Flux2Klein4B => "flux2-klein-4b".into(),
            Architecture::Flux2Klein9B => "flux2-klein-9b".into(),
            Architecture::Flux1Dev => "flux1-dev".into(),
            Architecture::Flux1Schnell => "flux1-schnell".into(),
            Architecture::Sd35Large => "sd35-large".into(),
            Architecture::Sd35Medium => "sd35-medium".into(),
            Architecture::Qwen3 => "qwen3".into(),
            Architecture::Llama => "llama".into(),
            Architecture::DeepSeek => "deepseek".into(),
            Architecture::Gemma => "gemma".into(),
            Architecture::Mistral3 => "mistral3".into(),
            Architecture::T5 => "t5".into(),
            Architecture::ClipL => "clip-l".into(),
            Architecture::OpenClip => "open-clip".into(),
            Architecture::Sdxl => "sdxl".into(),
            Architecture::Sd15 => "sd15".into(),
            Architecture::Unknown(s) => s.clone(),
        }
    }
}

/// Detect the architecture from a weights source (already opened archive / GGUF) by sniffing keys.
///
/// Returns an `Architecture` best-effort. Callers can layer extra heuristics (e.g. reading
/// `config.json` for model_type) on top; this table is the shape-based fallback.
pub fn detect_architecture(src: &dyn WeightsSource) -> Architecture {
    let keys = src.keys();
    let has = |needle: &str| keys.iter().any(|k| k.contains(needle));

    // --- Flux family ---------------------------------------------------------
    if has("double_stream_modulation") || has("img_attn.norm.key_norm.scale") {
        // Klein and Dev both carry these; disambiguate on guidance + single-block count.
        let has_guidance = has("guidance_in") || has("time_guidance_embed");
        let max_single = count_blocks(&keys, &["single_blocks.", "single_transformer_blocks."]);
        let max_double = count_blocks(&keys, &["double_blocks.", "transformer_blocks."]);
        if has_guidance && max_single > 40 {
            return Architecture::Flux2Dev;
        }
        if max_double == 8 && max_single == 24 {
            return Architecture::Flux2Klein9B;
        }
        return Architecture::Flux2Klein4B;
    }
    if has("guidance_in") || has("time_guidance_embed") {
        let max_single = count_blocks(&keys, &["single_blocks.", "single_transformer_blocks."]);
        return if max_single > 40 { Architecture::Flux2Dev } else { Architecture::Flux1Dev };
    }
    if has("double_blocks.") || has("single_blocks.") {
        // Flux.1 (no guidance embedder): distinguish Schnell/Dev via block count.
        let max_single = count_blocks(&keys, &["single_blocks.", "single_transformer_blocks."]);
        return if max_single > 20 { Architecture::Flux1Dev } else { Architecture::Flux1Schnell };
    }

    // --- SD3 / SD3.5 (BFL native: joint_blocks + x_embedder/context_embedder) ---
    if has("joint_blocks.") || has("x_embedder.proj") {
        let max_joint = count_blocks(&keys, &["joint_blocks.", "model.diffusion_model.joint_blocks."]);
        return if max_joint > 30 { Architecture::Sd35Large } else { Architecture::Sd35Medium };
    }

    // --- CausalLM / Text Models -----------------------------------------------
    if has("model.embed_tokens.weight") || has("embed_tokens.weight") || has("token_embd.weight") {
        if has("gemma") || has("pre_feedforward_layernorm") {
            return Architecture::Gemma;
        }
        if has("deepseek") || has("mla") {
            return Architecture::DeepSeek;
        }
        if has("self_attn.q_proj.weight") && (has("layers.0.") || has("model.layers.0.")) {
            if has("q_norm") || has("attn_q_norm") {
                return Architecture::Qwen3;
            }
            return Architecture::Llama;
        }
        if has("blk.0.attn_q.weight") {
            if has("attn_q_norm") || has("ssm_a") {
                return Architecture::Qwen3;
            }
            return Architecture::Llama;
        }
        if has("language_model.model.layers.0.") || has("model.layers.0.") {
            return Architecture::Mistral3;
        }
        return Architecture::Llama;
    }
    if has("encoder.layers.") && has("decoder.layers.") {
        return Architecture::T5;
    }
    // --- SDXL / SD1.5 --------------------------------------------------------
    // SDXL carries `conditioner.embedders.*` BEFORE the text-model keys, so it MUST be checked first,
    // otherwise the embedded CLIP-L / OpenCLIP text encoders shadow the SDXL family detection.
    if has("conditioner.embedders") || has("add_embedding") || (has("cross_attention_dim") && has("time_embedding")) {
        return Architecture::Sdxl;
    }
    if has("model.diffusion_model.input_blocks.") {
        return Architecture::Sd15;
    }

    if has("text_model.encoder.layers.") && has("text_projection.weight") {
        return Architecture::OpenClip;
    }
    if has("text_model.encoder.layers.") {
        return Architecture::ClipL;
    }

    Architecture::Unknown("unknown".into())
}

/// Detect from an optional `config.json` `model_type` string (HF convention), if present.
pub fn detect_from_model_type(model_type: Option<&str>) -> Result<Architecture> {
    let t = model_type.unwrap_or_default().to_ascii_lowercase();
    if t.is_empty() {
        return Ok(Architecture::Unknown("unset".into()));
    }
    let arch = match t.as_str() {
        "flux2-dev" | "flux2dev" => Architecture::Flux2Dev,
        "flux2-klein-4b" | "flux2klein4b" => Architecture::Flux2Klein4B,
        "flux2-klein-9b" | "flux2klein9b" => Architecture::Flux2Klein9B,
        "flux2" => Architecture::Flux2Dev,
        "flux1" | "flux" => Architecture::Flux1Dev,
        "sd3" | "sd35" | "sd3.5" | "stable-diffusion-3" | "stable-diffusion-3.5" => Architecture::Sd35Large,
        "qwen3" | "qwen2" | "qwen" => Architecture::Qwen3,
        "llama" | "llama3" | "llama2" => Architecture::Llama,
        "deepseek" | "deepseek_v2" | "deepseek_v3" => Architecture::DeepSeek,
        "gemma" | "gemma2" | "gemma3" => Architecture::Gemma,
        "mistral" | "mistral3" => Architecture::Mistral3,
        "t5" | "t5xxl" => Architecture::T5,
        "clip" | "clip_l" | "clip-l" => Architecture::ClipL,
        "open_clip" | "openclip" => Architecture::OpenClip,
        "sdxl" | "stable-diffusion-xl" => Architecture::Sdxl,
        "sd15" | "stable-diffusion-v1" => Architecture::Sd15,
        other => {
            return Err(LuminaError::UnknownArchitecture { family: other.to_string() });
        }
    };
    Ok(arch)
}

fn count_blocks(keys: &[String], prefixes: &[&str]) -> usize {
    let mut max_s = 0;
    for k in keys {
        for p in prefixes {
            if let Some(rest) = k.strip_prefix(p) {
                if let Some(idx_str) = rest.split('.').next() {
                    if let Ok(idx) = idx_str.parse::<usize>() {
                        max_s = max_s.max(idx + 1);
                    }
                }
            }
        }
    }
    max_s
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn model_type_strings_map_to_architectures() {
        assert_eq!(detect_from_model_type(Some("qwen3")).unwrap(), Architecture::Qwen3);
        assert_eq!(detect_from_model_type(Some("mistral")).unwrap(), Architecture::Mistral3);
        assert_eq!(detect_from_model_type(Some("t5")).unwrap(), Architecture::T5);
        assert_eq!(detect_from_model_type(Some("flux2-dev")).unwrap(), Architecture::Flux2Dev);
        assert_eq!(detect_from_model_type(Some("sdxl")).unwrap(), Architecture::Sdxl);
        assert_eq!(detect_from_model_type(Some("sd15")).unwrap(), Architecture::Sd15);
    }

    #[test]
    fn unknown_model_type_is_error() {
        assert!(detect_from_model_type(Some("gpt4-ultra")).is_err());
    }

    #[test]
    fn slug_and_kind_are_consistent() {
        assert_eq!(Architecture::Flux2Dev.slug(), "flux2-dev");
        assert_eq!(Architecture::Flux2Dev.model_kind(), ModelKind::Diffusion);
        assert_eq!(Architecture::Qwen3.model_kind(), ModelKind::TextEncoder);
    }

    #[test]
    fn shape_sniff_detects_flux2_dev() {
        // Fake keys that mimic a Flux.2-Dev checkpoint (guidance embed + 48 single / 8 double).
        let keys: Vec<String> = {
            let mut v = vec![
                "double_stream_modulation_img.lin.weight".to_string(),
                "guidance_in.in_layer.weight".to_string(),
            ];
            for i in 0..48 { v.push(format!("single_blocks.{i}.linear1.weight")); }
            for i in 0..8 { v.push(format!("double_blocks.{i}.img_attn.qkv.weight")); }
            v
        };
        let src = FakeSource { keys };
        assert_eq!(detect_architecture(&src), Architecture::Flux2Dev);
    }

    #[test]
    fn shape_sniff_sdxl_is_not_shadowed_by_embedders() {
        // SDXL ships its CLIP-L/OpenCLIP behind `conditioner.embedders.*`; the family must still win
        // over the text-encoder heuristics.
        let keys: Vec<String> = vec![
            "model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_q.weight".to_string(),
            "conditioner.embedders.0.transformer.text_model.encoder.layers.0.self_attn.v_proj.bias".to_string(),
            "conditioner.embedders.1.model.transformer.resblocks.0.attn.in_proj_weight".to_string(),
        ];
        let src = FakeSource { keys };
        assert_eq!(detect_architecture(&src), Architecture::Sdxl);
    }

    #[test]
    fn shape_sniff_detects_sd35() {
        let mut keys = vec![
            "model.diffusion_model.x_embedder.proj.weight".to_string(),
            "model.diffusion_model.context_embedder.weight".to_string(),
        ];
        for i in 0..38 { keys.push(format!("model.diffusion_model.joint_blocks.{i}.x_block.attn.qkv.weight")); }
        assert_eq!(detect_architecture(&FakeSource { keys }), Architecture::Sd35Large);

        let mut med = vec!["model.diffusion_model.x_embedder.proj.weight".to_string()];
        for i in 0..24 { med.push(format!("model.diffusion_model.joint_blocks.{i}.x_block.attn.qkv.weight")); }
        assert_eq!(detect_architecture(&FakeSource { keys: med }), Architecture::Sd35Medium);
    }

    #[test]
    fn shape_sniff_detects_qwen3() {
        let keys: Vec<String> = vec![
            "model.embed_tokens.weight".to_string(),
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            "model.layers.0.mlp.gate_proj.weight".to_string(),
        ];
        let src = FakeSource { keys };
        assert_eq!(detect_architecture(&src), Architecture::Qwen3);
    }

    /// Minimal `WeightsSource` stub for shape-based detection tests.
    struct FakeSource {
        keys: Vec<String>,
    }

    impl crate::weights::WeightsSource for FakeSource {
        fn get_tensor(&self, _name: &str, _device: &Device, _dtype: DType) -> crate::error::Result<candle_core::Tensor> {
            Err(crate::error::LuminaError::MissingWeight("stub".into()))
        }
        fn contains(&self, name: &str) -> bool { self.keys.iter().any(|k| k == name) }
        fn raw_info(&self, _name: &str) -> Option<(safetensors::Dtype, Vec<usize>)> { None }
        fn keys(&self) -> Vec<String> { self.keys.clone() }
    }
}
