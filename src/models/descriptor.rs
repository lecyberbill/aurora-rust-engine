// [WFGY] Zone: SAFE | λ: 0.25 | Fallbacks: 0 | Action: Model descriptor — explicit checkpoint + VLM + VAE wiring per model family

use std::path::PathBuf;
use serde::Deserialize;
use crate::error::{LuminaError, Result};
use crate::models::config::Architecture;

/// Which text encoder to attach to a diffusion model (for families like Flux.2-Klein/Dev that do NOT
/// embed the conditioning text encoder). Self-contained models (Flux.1, SDXL) leave this `None` and
/// use their embedded encoders.
#[derive(Debug, Clone)]
pub enum TextEncoderSpec {
    /// Qwen3 encoder from a single `.safetensors` file or a shards directory.
    Qwen3 { path: PathBuf },
    /// Mistral-3 VLM from a directory of shards (Flux.2-Dev official).
    Mistral3 { dir: PathBuf },
    /// T5-XXL (rarely external — Flux.1 embeds it) from a safetensors file.
    T5 { path: PathBuf },
    /// SD 3.5 uses **three** text encoders at once: CLIP-L, CLIP-G (OpenCLIP bigG) and T5-XXL.
    Sd35 { clip_l: PathBuf, clip_g: PathBuf, t5: PathBuf },
}

/// Explicit, per-model wiring: the DiT/UNet checkpoint plus the external VLM + VAE it needs
/// (if any). This is what a platform uses to *control* model loading (and unload on switch),
/// rather than relying on auto-detection of embedded encoders.
#[derive(Debug, Clone)]
pub struct ModelDescriptor {
    pub id: String,
    pub checkpoint: PathBuf,
    /// Optional hint; when `None` the architecture is sniffed from the checkpoint.
    pub family: Option<Architecture>,
    pub text_encoder: Option<TextEncoderSpec>,
    /// Path to a 32-channel Flux VAE (`flux2-vae.safetensors`) for Flux.2 families that don't embed
    /// one. `None` means the VAE is embedded (Flux.1) or handled by the SDXL pipeline.
    pub vae: Option<PathBuf>,
}

impl ModelDescriptor {
    /// A self-contained model (SDXL, Flux.1): no external VLM/VAE.
    pub fn standalone(id: impl Into<String>, checkpoint: impl Into<PathBuf>) -> Self {
        Self { id: id.into(), checkpoint: checkpoint.into(), family: None, text_encoder: None, vae: None }
    }

    /// A Flux.2-Klein/Dev model: attach the matching VLM and the shared 32-ch Flux VAE.
    pub fn flux2(
        id: impl Into<String>,
        checkpoint: impl Into<PathBuf>,
        text_encoder: TextEncoderSpec,
        vae: impl Into<PathBuf>,
    ) -> Self {
        Self {
            id: id.into(),
            checkpoint: checkpoint.into(),
            family: None,
            text_encoder: Some(text_encoder),
            vae: Some(vae.into()),
        }
    }

    /// An SD 3.5 model: attach CLIP-L + CLIP-G + T5-XXL and the 16-ch SD3 VAE.
    pub fn sd35(
        id: impl Into<String>,
        checkpoint: impl Into<PathBuf>,
        clip_l: impl Into<PathBuf>,
        clip_g: impl Into<PathBuf>,
        t5: impl Into<PathBuf>,
        vae: impl Into<PathBuf>,
    ) -> Self {
        Self {
            id: id.into(),
            checkpoint: checkpoint.into(),
            family: None,
            text_encoder: Some(TextEncoderSpec::Sd35 {
                clip_l: clip_l.into(),
                clip_g: clip_g.into(),
                t5: t5.into(),
            }),
            vae: Some(vae.into()),
        }
    }
}

// ---------------------------------------------------------------------------
// JSON model list ("config.json") — a declarative, reusable alternative to
// hard-coding checkpoint paths in an application (studio, server, CLI).
// ---------------------------------------------------------------------------

/// A JSON list of models, e.g.:
///
/// ```json
/// {
///   "models": [
///     { "id": "sdxl", "label": "SDXL", "family": "sdxl",
///       "checkpoint": "G:/models/checkpoints/sdxl.safetensors" },
///     { "id": "sd35", "label": "SD 3.5 Large", "family": "sd35",
///       "checkpoint": "G:/models/SD3/sd3.5_large.safetensors",
///       "text_encoder": { "kind": "sd35", "clip_l": "…", "clip_g": "…", "t5": "…" },
///       "vae": "G:/models/vae/sd3_vae.safetensors" }
///   ]
/// }
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct ModelDescriptorFile {
    pub models: Vec<ModelDescriptorEntry>,
}

/// One entry in a [`ModelDescriptorFile`]. `label` is presentation-only.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelDescriptorEntry {
    pub id: String,
    #[serde(default)]
    pub label: Option<String>,
    /// Architecture hint slug (`"sdxl"`, `"sd35"`, `"flux1"`, `"flux2-klein-4b"`, `"sd15"`, ...).
    /// `None` (or absent) means the family is sniffed from the checkpoint.
    #[serde(default)]
    pub family: Option<String>,
    pub checkpoint: PathBuf,
    #[serde(default)]
    pub text_encoder: Option<TextEncoderConfig>,
    #[serde(default)]
    pub vae: Option<PathBuf>,
    /// Optional per-model generation defaults (steps / guidance / size / negative prompt).
    #[serde(default)]
    pub defaults: ModelDefaults,
}

/// Per-model generation defaults carried by the config (all optional). A UI can apply them when
/// switching models instead of hard-coding slider values.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct ModelDefaults {
    #[serde(default)]
    pub steps: Option<usize>,
    #[serde(default)]
    pub guidance: Option<f64>,
    #[serde(default)]
    pub width: Option<usize>,
    #[serde(default)]
    pub height: Option<usize>,
    #[serde(default)]
    pub negative_prompt: Option<String>,
}

/// A fully-resolved model entry: presentation label + descriptor + generation defaults.
#[derive(Debug, Clone)]
pub struct ResolvedModel {
    pub id: String,
    pub label: String,
    pub descriptor: ModelDescriptor,
    pub defaults: ModelDefaults,
}

/// Serde mirror of [`TextEncoderSpec`], discriminated by a `"kind"` field.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TextEncoderConfig {
    Qwen3 { path: PathBuf },
    Mistral3 { dir: PathBuf },
    T5 { path: PathBuf },
    Sd35 { clip_l: PathBuf, clip_g: PathBuf, t5: PathBuf },
}

impl TextEncoderConfig {
    pub fn to_spec(&self) -> TextEncoderSpec {
        match self {
            Self::Qwen3 { path } => TextEncoderSpec::Qwen3 { path: path.clone() },
            Self::Mistral3 { dir } => TextEncoderSpec::Mistral3 { dir: dir.clone() },
            Self::T5 { path } => TextEncoderSpec::T5 { path: path.clone() },
            Self::Sd35 { clip_l, clip_g, t5 } => TextEncoderSpec::Sd35 {
                clip_l: clip_l.clone(),
                clip_g: clip_g.clone(),
                t5: t5.clone(),
            },
        }
    }
}

impl ModelDescriptorEntry {
    /// Build a [`ModelDescriptor`] from this entry, resolving the optional family hint slug.
    pub fn to_descriptor(&self) -> Result<ModelDescriptor> {
        let family = match &self.family {
            Some(s) => Some(crate::models::config::detect_from_model_type(Some(s))?),
            None => None,
        };
        Ok(ModelDescriptor {
            id: self.id.clone(),
            checkpoint: self.checkpoint.clone(),
            family,
            text_encoder: self.text_encoder.as_ref().map(TextEncoderConfig::to_spec),
            vae: self.vae.clone(),
        })
    }
}

impl ModelDescriptorFile {
    /// Parse a models `config.json` string.
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(|e| LuminaError::Config(format!("models config.json: {e}")))
    }

    /// Load a models `config.json` from disk.
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .map_err(|e| LuminaError::Config(format!("cannot read {}: {e}", path.display())))?;
        Self::from_json(&text)
    }

    /// Resolve every entry to `(label, ModelDescriptor)` (label falls back to the id).
    pub fn descriptors(&self) -> Result<Vec<(String, ModelDescriptor)>> {
        self.models
            .iter()
            .map(|e| Ok((e.label.clone().unwrap_or_else(|| e.id.clone()), e.to_descriptor()?)))
            .collect()
    }

    /// Resolve every entry, keeping the id, presentation label and per-model generation defaults.
    pub fn resolve(&self) -> Result<Vec<ResolvedModel>> {
        self.models
            .iter()
            .map(|e| {
                Ok(ResolvedModel {
                    id: e.id.clone(),
                    label: e.label.clone().unwrap_or_else(|| e.id.clone()),
                    descriptor: e.to_descriptor()?,
                    defaults: e.defaults.clone(),
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_model_list_config() {
        let json = r#"{"models":[
            {"id":"sdxl","label":"SDXL","family":"sdxl","checkpoint":"a.safetensors"},
            {"id":"sd35","label":"SD 3.5","family":"sd35","checkpoint":"b.safetensors",
             "text_encoder":{"kind":"sd35","clip_l":"l","clip_g":"g","t5":"t"},"vae":"v"}
        ]}"#;
        let file = ModelDescriptorFile::from_json(json).unwrap();
        let ds = file.descriptors().unwrap();
        assert_eq!(ds.len(), 2);
        assert_eq!(ds[0].0, "SDXL");
        assert_eq!(ds[0].1.family, Some(Architecture::Sdxl));
        match &ds[1].1.text_encoder {
            Some(TextEncoderSpec::Sd35 { clip_l, t5, .. }) => {
                assert_eq!(clip_l.to_string_lossy(), "l");
                assert_eq!(t5.to_string_lossy(), "t");
            }
            other => panic!("expected Sd35 spec, got {other:?}"),
        }
        assert_eq!(ds[1].1.vae.as_deref(), Some(std::path::Path::new("v")));
    }

    #[test]
    fn unknown_family_slug_is_error() {
        let json = r#"{"models":[{"id":"x","family":"gpt42","checkpoint":"x.safetensors"}]}"#;
        let file = ModelDescriptorFile::from_json(json).unwrap();
        assert!(file.descriptors().is_err());
    }

    #[test]
    fn parse_model_defaults() {
        let json = r#"{"models":[
            {"id":"sdxl","label":"SDXL","family":"sdxl","checkpoint":"a.safetensors",
             "defaults":{"steps":25,"guidance":7.0,"width":1024,"height":1024}},
            {"id":"sd35","label":"SD 3.5","family":"sd35","checkpoint":"b.safetensors",
             "defaults":{"steps":28,"guidance":3.5,"negative_prompt":"blurry"}}
        ]}"#;
        let file = ModelDescriptorFile::from_json(json).unwrap();
        let ms = file.resolve().unwrap();
        assert_eq!(ms.len(), 2);
        assert_eq!(ms[0].id, "sdxl");
        assert_eq!(ms[0].label, "SDXL");
        assert_eq!(ms[0].defaults.steps, Some(25));
        assert_eq!(ms[0].defaults.guidance, Some(7.0));
        assert_eq!(ms[0].defaults.width, Some(1024));
        assert_eq!(ms[0].defaults.height, Some(1024));
        assert_eq!(ms[0].defaults.negative_prompt, None);
        assert_eq!(ms[1].defaults.steps, Some(28));
        assert_eq!(ms[1].defaults.guidance, Some(3.5));
        assert_eq!(ms[1].defaults.negative_prompt.as_deref(), Some("blurry"));
        assert_eq!(ms[1].descriptor.family, Some(Architecture::Sd35Large));
    }

    #[test]
    fn defaults_are_optional() {
        let json = r#"{"models":[{"id":"x","family":"sdxl","checkpoint":"x.safetensors"}]}"#;
        let ms = ModelDescriptorFile::from_json(json).unwrap().resolve().unwrap();
        assert_eq!(ms[0].defaults.steps, None);
        assert_eq!(ms[0].defaults.negative_prompt, None);
    }
}
