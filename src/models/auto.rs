// [WFGY] Zone: SAFE | λ: 0.30 | Fallbacks: 1 (config.json missing -> shape sniff) | Action: `transformers`-style AutoModel/AutoConfig/AutoTokenizer (Candle port)

use candle_core::{Device, DType};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::error::{LuminaError, Result};
use crate::hub::ModelHub;
use crate::weights::SafeTensorsArchive;

use super::common::AnyModel;
use super::config::{detect_architecture, Architecture};
use super::image::DiffusionModel;
use super::text::TextModel;

/// The "Auto" entry point. Callers say *what* they want (a repo id or a local checkpoint path) and
/// get back a ready-to-use model object, mirroring `transformers.AutoModel.from_pretrained`.
pub struct AutoModel;

/// Result of a resolved model, boxed so the platform never names a concrete pipeline type.
pub type BoxedModel = Arc<Mutex<dyn AnyModel>>;

/// A `config.json`-style model type hint, used to resolve a repo without loading weights.
#[derive(Debug, Clone)]
pub struct ModelLoadConfig {
    pub repo: String,
    pub revision: Option<String>,
    pub device: Device,
    pub dtype: DType,
    /// Relative path to `config.json` inside the repo (default `"config.json"`).
    pub config_file: String,
    /// Preferred set of weight shard files to resolve (empty = auto from `model.safetensors.index.json`).
    pub shard_files: Vec<String>,
}

impl ModelLoadConfig {
    pub fn new(repo: impl Into<String>, device: Device, dtype: DType) -> Self {
        Self {
            repo: repo.into(),
            revision: None,
            device,
            dtype,
            config_file: "config.json".into(),
            shard_files: Vec::new(),
        }
    }
}

impl AutoModel {
    /// Resolve a local checkpoint path (folder of shards or single `.safetensors`/`.gguf`), open it,
    /// sniff the architecture and build the best-matching model object.
    pub fn from_local(path: impl AsRef<Path>, device: Device, dtype: DType) -> Result<BoxedModel> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(LuminaError::ModelNotFound(format!("{}", path.display())));
        }

        // ---- Open the weights source (detect format by extension) -----------------
        let archive: Arc<dyn crate::weights::WeightsSource> = if path.is_dir() {
            Arc::new(SafeTensorsArchive::open_shards_dir(path)?)
        } else {
            let lower = path.to_string_lossy().to_lowercase();
            if lower.ends_with(".gguf") {
                Arc::new(crate::gguf::GgufWeights::open(path)?)
            } else {
                Arc::new(SafeTensorsArchive::open(path)?)
            }
        };

        let arch = detect_architecture(&*archive);
        Self::build(arch, path.to_path_buf(), device, dtype)
    }

    /// Resolve a Hugging Face (or mirror) repo id, download the weights via [`ModelHub`], detect the
    /// architecture and build the model. When a `config.json` ships, `model_type` is authoritative.
    ///
    /// This is synchronous (like `hf_hub`'s sync `Api`); a caller wanting concurrency wraps the call
    /// in `tokio::task::spawn_blocking`. Kept simple so the engine stays a plain library.
    pub fn from_pretrained(hub: &ModelHub, cfg: ModelLoadConfig) -> Result<BoxedModel> {
        let repo = cfg.repo.clone();

        // Try to read model_type from config.json so we don't need to load all weights.
        let arch_from_config = read_model_type(hub, &repo, cfg.revision.as_deref(), &cfg.config_file)?;

        // Resolve a local folder of weights (all shards or a single file).
        let folder = if cfg.shard_files.is_empty() {
            hub.resolve_hf_repo(&repo, &auto_shard_names(hub, &repo, cfg.revision.as_deref())?, cfg.revision.as_deref())?
        } else {
            hub.resolve_hf_repo(&repo, &cfg.shard_files, cfg.revision.as_deref())?
        };

        let archive: Arc<dyn crate::weights::WeightsSource> =
            Arc::new(SafeTensorsArchive::open_shards_dir(&folder)?);

        let arch = match arch_from_config {
            Some(a) => a,
            None => detect_architecture(&*archive),
        };

        Self::build(arch, folder, cfg.device, cfg.dtype)
    }

    /// Build a concrete model object from a detected architecture.
    fn build(arch: Architecture, weights: PathBuf, device: Device, dtype: DType) -> Result<BoxedModel> {
        let boxed: BoxedModel = match arch {
            // ---- Diffusion ------------------------------------------------------
            Architecture::Flux2Dev | Architecture::Flux2Klein4B | Architecture::Flux2Klein9B
            | Architecture::Flux1Dev | Architecture::Flux1Schnell
            | Architecture::Sdxl | Architecture::Sd15 => {
                let diff = super::auto::build_diffusion(&arch, &weights, &device, dtype)?;
                Arc::new(Mutex::new(diff))
            }
            // ---- Text encoders --------------------------------------------------
            Architecture::Qwen3 | Architecture::Mistral3 | Architecture::T5
            | Architecture::ClipL | Architecture::OpenClip => {
                let text = super::auto::build_text(&arch, &weights, &device, dtype)?;
                Arc::new(Mutex::new(text))
            }
            Architecture::Unknown(name) => {
                return Err(LuminaError::UnknownArchitecture { family: name });
            }
        };
        Ok(boxed)
    }
}

fn build_diffusion(
    arch: &Architecture,
    weights: &Path,
    device: &Device,
    dtype: DType,
) -> Result<DiffusionModel> {
    let id = arch.slug();
    match arch {
        Architecture::Flux2Dev
        | Architecture::Flux2Klein4B
        | Architecture::Flux2Klein9B
        | Architecture::Flux1Dev
        | Architecture::Flux1Schnell => {
            let mut pipeline = if weights.to_string_lossy().to_lowercase().ends_with(".gguf") {
                crate::pipelines::FluxPipeline::from_gguf_dtype(weights, device.clone(), dtype)?
            } else {
                crate::pipelines::FluxPipeline::from_single_file_streaming(weights, device.clone())?
            };
            pipeline.enable_flash_attn();
            Ok(DiffusionModel::flux(id, arch.slug(), pipeline))
        }
        Architecture::Sdxl => {
            let pipeline = crate::pipelines::StableDiffusionXLPipeline::from_single_file(weights, device.clone())?;
            Ok(DiffusionModel::sdxl(id, pipeline))
        }
        Architecture::Sd15 => {
            let pipeline = <crate::pipelines::StableDiffusionPipeline as crate::traits::TextToImagePipeline>::from_safetensors(weights, device)?;
            Ok(DiffusionModel::sd15(id, pipeline))
        }
        _ => Err(LuminaError::UnsupportedOp(format!("diffusion build for {}", arch.slug()))),
    }
}

fn build_text(
    arch: &Architecture,
    weights: &Path,
    device: &Device,
    dtype: DType,
) -> Result<TextModel> {
    let id = arch.slug();
    match arch {
        Architecture::Qwen3 => {
            // Open shards if a dir, else a single file.
            let archive: Arc<dyn crate::weights::WeightsSource> = if weights.is_dir() {
                Arc::new(SafeTensorsArchive::open_shards_dir(weights)?)
            } else {
                Arc::new(SafeTensorsArchive::open(weights)?)
            };
            let enc = crate::text::Qwen3TextEncoder::from_archive(&*archive, None, &Device::Cpu, dtype)?;
            Ok(TextModel::qwen(id, enc))
        }
        Architecture::Mistral3 => {
            let enc = if weights.is_dir() {
                crate::text::Mistral3TextEncoder::from_dir(weights, None, Device::Cpu, dtype)?
            } else {
                crate::text::Mistral3TextEncoder::from_safetensors(weights, None, Device::Cpu, dtype)?
            };
            Ok(TextModel::mistral(id, enc))
        }
        Architecture::T5 => {
            let archive = if weights.is_dir() {
                SafeTensorsArchive::open_shards_dir(weights)?
            } else {
                SafeTensorsArchive::open(weights)?
            };
            let mut tensors = std::collections::HashMap::new();
            for key in archive.keys() {
                if let Ok(t) = archive.get_tensor(&key, device, dtype) {
                    tensors.insert(key.to_string(), t);
                }
            }
            let vb = candle_nn::VarBuilder::from_tensors(tensors, dtype, device);
            let iter = crate::text::T5TextEncoder::new(vb, None)?;
            Ok(TextModel::t5(id, iter))
        }
        _ => Err(LuminaError::UnsupportedOp(format!("text build for {}", arch.slug()))),
    }
}

/// Read `model_type` from a repo's `config.json` without loading weights.
fn read_model_type(
    hub: &ModelHub,
    repo: &str,
    revision: Option<&str>,
    config_file: &str,
) -> Result<Option<Architecture>> {
    let cfg_path = match hub.resolve_hf(repo, config_file, revision) {
        Ok(p) => p,
        Err(_) => return Ok(None),
    };
    let text = match std::fs::read_to_string(&cfg_path) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };
    let model_type = parse_model_type_from_json(&text);
    match model_type {
        Some(t) => super::config::detect_from_model_type(Some(&t)).map(Some),
        None => Ok(None),
    }
}

/// Minimal, dependency-free `model_type` extraction from `config.json`.
fn parse_model_type_from_json(json: &str) -> Option<String> {
    let key = "\"model_type\"";
    let pos = json.find(key)? + key.len();
    let rest = &json[pos..];
    let colon = rest.find(':')? + 1;
    let after = rest[colon..].trim_start();
    let value = after.strip_prefix('"')?;
    let end = value.find('"')?;
    Some(value[..end].to_string())
}

/// Auto-discover shard files from a repo (index JSON or the common `model-0000N-of-0000M.safetensors`).
fn auto_shard_names(hub: &ModelHub, repo: &str, revision: Option<&str>) -> Result<Vec<String>> {
    // Try the safetensors index first.
    for idx in ["model.safetensors.index.json", "model.safetensors"] {
        if let Ok(_) = hub.resolve_hf(repo, idx, revision) {
            return Ok(vec![idx.to_string()]);
        }
    }
    // Fall back to the standard multi-shard naming.
    let mut names = Vec::new();
    let mut m = 1usize;
    loop {
        let f = format!("model-{m:05}-of-00001.safetensors");
        if hub.resolve_hf(repo, &f, revision).is_err() {
            break;
        }
        names.push(f);
        m += 1;
        if m > 64 { break; }
    }
    if names.is_empty() {
        Err(LuminaError::ModelNotFound(format!(
            "no safetensors weight files found in repo {repo}"
        )))
    } else {
        Ok(names)
    }
}
