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

    /// Build a model from an explicit [`ModelDescriptor`], attaching any external VLM/VAE the family
    /// needs (e.g. Qwen3 + 32-ch VAE for Flux.2-Klein-4B). This is the control-plane entry point a
    /// platform uses so it can load / swap / unload models by descriptor.
    pub fn from_descriptor(
        desc: &super::descriptor::ModelDescriptor,
        device: Device,
        dtype: DType,
    ) -> Result<BoxedModel> {
        let arch = match &desc.family {
            Some(a) => a.clone(),
            None => {
                let archive: Arc<dyn crate::weights::WeightsSource> = if desc.checkpoint.is_dir() {
                    Arc::new(SafeTensorsArchive::open_shards_dir(&desc.checkpoint)?)
                } else if desc.checkpoint.to_string_lossy().to_lowercase().ends_with(".gguf") {
                    Arc::new(crate::gguf::GgufWeights::open(&desc.checkpoint)?)
                } else {
                    Arc::new(SafeTensorsArchive::open(&desc.checkpoint)?)
                };
                detect_architecture(&*archive)
            }
        };

        let boxed: BoxedModel = match arch {
            Architecture::Flux2Dev
            | Architecture::Flux2Klein4B
            | Architecture::Flux2Klein9B
            | Architecture::Flux1Dev
            | Architecture::Flux1Schnell => {
                let mut pipeline = if desc.checkpoint.to_string_lossy().to_lowercase().ends_with(".gguf") {
                    crate::pipelines::FluxPipeline::from_gguf_dtype(&desc.checkpoint, device.clone(), dtype)?
                } else {
                    crate::pipelines::FluxPipeline::from_single_file_streaming(&desc.checkpoint, device.clone())?
                };
                pipeline.enable_flash_attn();

                // Attach external text encoder (Flux.2-Klein/Dev don't embed one).
                if let Some(spec) = &desc.text_encoder {
                    match spec {
                        super::descriptor::TextEncoderSpec::Qwen3 { path } => {
                            let archive: Arc<dyn crate::weights::WeightsSource> = if path.is_dir() {
                                Arc::new(SafeTensorsArchive::open_shards_dir(path)?)
                            } else {
                                Arc::new(SafeTensorsArchive::open(path)?)
                            };
                            let enc = crate::text::Qwen3TextEncoder::from_archive(
                                &*archive,
                                Some(std::path::Path::new("qwen_tokenizer.json")),
                                &device,
                                dtype,
                            )?;
                            pipeline.set_qwen3(enc);
                        }
                        super::descriptor::TextEncoderSpec::Mistral3 { dir } => {
                            let enc = crate::text::Mistral3TextEncoder::from_dir(
                                dir,
                                Some(std::path::Path::new("mistral_tokenizer.json")),
                                device.clone(),
                                dtype,
                            )?;
                            pipeline.set_mistral(enc);
                        }
                        super::descriptor::TextEncoderSpec::T5 { path: _ } => {
                            // Flux.1 embeds T5-XXL and wires it internally, so an external T5 is
                            // intentionally not wired here. Keep the variant for completeness.
                        }
                        super::descriptor::TextEncoderSpec::Sd35 { .. } => {
                            return Err(LuminaError::Config(
                                "TextEncoderSpec::Sd35 used with a Flux model".into(),
                            ));
                        }
                    }
                }

                // Attach the 32-channel Flux VAE if given (Flux.2 needs the external flux2-vae).
                if let Some(vae_path) = &desc.vae {
                    let vae_archive = if vae_path.is_dir() {
                        SafeTensorsArchive::open_shards_dir(vae_path)?
                    } else {
                        SafeTensorsArchive::open(vae_path)?
                    };
                    let vae_router = crate::weights::WeightRouter::new(&vae_archive, device.clone(), dtype);
                    let vae_vb = vae_router.vae_var_builder()?;
                    let decoder = crate::diffusion::vae_flux::FluxVaeDecoder::new(vae_vb.clone())?;
                    let encoder = crate::diffusion::vae_flux::FluxVaeEncoder::new(vae_vb)?;
                    pipeline.set_vae(decoder);
                    pipeline.set_vae_encoder(encoder);
                }

                Arc::new(Mutex::new(DiffusionModel::flux(desc.id.clone(), arch.slug(), pipeline)))
            }
            Architecture::Sdxl => {
                let mem = desc
                    .memory
                    .as_ref()
                    .map(|m| m.to_pipeline_config())
                    .unwrap_or_default();
                let pipeline = crate::pipelines::StableDiffusionXLPipeline::from_single_file_with_config(
                    &desc.checkpoint,
                    device.clone(),
                    mem,
                )?;
                Arc::new(Mutex::new(DiffusionModel::sdxl(desc.id.clone(), pipeline)))
            }
            Architecture::Sd35Large | Architecture::Sd35Medium => {
                let mut pipeline = crate::pipelines::FluxPipeline::from_single_file_streaming(&desc.checkpoint, device.clone())?;
                pipeline.enable_flash_attn();

                // SD 3.5 needs three external text encoders (they are not embedded in the checkpoint).
                match &desc.text_encoder {
                    Some(super::descriptor::TextEncoderSpec::Sd35 { clip_l, clip_g, t5 }) => {
                        // T5-XXL must run in F32 (F16 overflows to NaN); CLIP-L/G in F16 on CPU.
                        let t5_enc = crate::text::T5TextEncoder::new(
                            vb_from_file(t5, &Device::Cpu, DType::F32)?,
                            Some(Path::new("t5xxl_tokenizer.json")),
                        )?;
                        let mut clip_l_enc = crate::text::ClipTextEncoder::new_sd15(
                            vb_from_file(clip_l, &Device::Cpu, DType::F16)?,
                        )?;
                        let _ = clip_l_enc.load_tokenizer("clip_tokenizer.json");
                        let mut clip_g_enc = crate::text::OpenClipTextEncoder::new_sdxl(
                            vb_from_file(clip_g, &Device::Cpu, DType::F16)?,
                        )?;
                        clip_g_enc.load_tokenizer("openclip_tokenizer.json")?;
                        pipeline.set_sd35_encoders(t5_enc, clip_l_enc);
                        pipeline.set_openclip_g(clip_g_enc);
                    }
                    other => {
                        return Err(LuminaError::Config(format!(
                            "SD3.5 needs TextEncoderSpec::Sd35 {{ clip_l, clip_g, t5 }} (got {})",
                            if other.is_some() { "another spec" } else { "None" }
                        )));
                    }
                }

                // 16-channel SD3 VAE.
                if let Some(vae_path) = &desc.vae {
                    let vae_archive = if vae_path.is_dir() {
                        SafeTensorsArchive::open_shards_dir(vae_path)?
                    } else {
                        SafeTensorsArchive::open(vae_path)?
                    };
                    let vae_router = crate::weights::WeightRouter::new(&vae_archive, device.clone(), dtype);
                    let vae_vb = vae_router.vae_var_builder()?;
                    pipeline.set_vae(crate::diffusion::vae_flux::FluxVaeDecoder::new(vae_vb)?);
                }

                Arc::new(Mutex::new(DiffusionModel::flux(desc.id.clone(), arch.slug(), pipeline)))
            }
            Architecture::Sd15 => {
                let pipeline = <crate::pipelines::StableDiffusionPipeline as crate::traits::TextToImagePipeline>::from_safetensors(&desc.checkpoint, &device)?;
                Arc::new(Mutex::new(DiffusionModel::sd15(desc.id.clone(), pipeline)))
            }
            other => {
                return Err(LuminaError::UnsupportedOp(format!("from_descriptor for {}", other.slug())));
            }
        };
        Ok(boxed)
    }

    /// Build a concrete model object from a detected architecture.
    fn build(arch: Architecture, weights: PathBuf, device: Device, dtype: DType) -> Result<BoxedModel> {
        let boxed: BoxedModel = match arch {
            // ---- Diffusion ------------------------------------------------------
            Architecture::ZImageTurbo
            | Architecture::Flux2Dev | Architecture::Flux2Klein4B | Architecture::Flux2Klein9B
            | Architecture::Flux1Dev | Architecture::Flux1Schnell
            | Architecture::Sd35Large | Architecture::Sd35Medium
            | Architecture::Sdxl | Architecture::Sd15 => {
                let diff = super::auto::build_diffusion(&arch, &weights, &device, dtype)?;
                Arc::new(Mutex::new(diff))
            }
            // ---- Text / CausalLM models -----------------------------------------
            Architecture::Qwen3 | Architecture::Mistral3 | Architecture::T5
            | Architecture::Llama | Architecture::DeepSeek | Architecture::Gemma
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
        Architecture::Sd35Large | Architecture::Sd35Medium => {
            // The transformer loads and detects; text encoders + VAE are attached by the caller via
            // `from_descriptor` (see `ModelDescriptor::sd35`). Generation without them errors clearly
            // in `encode_sd35`.
            let mut pipeline = crate::pipelines::FluxPipeline::from_single_file_streaming(weights, device.clone())?;
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
    let archive: Arc<dyn crate::weights::WeightsSource> = if weights.is_dir() {
        Arc::new(SafeTensorsArchive::open_shards_dir(weights)?)
    } else if weights.to_string_lossy().to_lowercase().ends_with(".gguf") {
        Arc::new(crate::gguf::GgufWeights::open(weights)?)
    } else {
        Arc::new(SafeTensorsArchive::open(weights)?)
    };

    match arch {
        Architecture::Llama | Architecture::DeepSeek | Architecture::Gemma | Architecture::Qwen3 | Architecture::Mistral3 => {
            let config = crate::models::text::CausalLMConfig::from_weights(&*archive)?;
            let tokenizer = resolve_tokenizer(weights, arch);
            let pipeline = crate::models::text::CausalLMPipeline::new(
                archive,
                config,
                tokenizer,
                device.clone(),
                dtype,
            );
            Ok(TextModel::causal_lm(id, arch.slug(), pipeline))
        }
        Architecture::T5 => {
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

fn resolve_tokenizer(weights: &Path, arch: &Architecture) -> Option<tokenizers::Tokenizer> {
    // 1. Try local tokenizer.json in model dir or alongside file
    let local_dir = if weights.is_dir() {
        weights.to_path_buf()
    } else {
        weights.parent().unwrap_or(Path::new(".")).to_path_buf()
    };
    let local_tok = local_dir.join("tokenizer.json");
    if local_tok.exists() {
        if let Ok(t) = tokenizers::Tokenizer::from_file(&local_tok) {
            return Some(t);
        }
    }

    // 2. Try standard root cache tokenizers
    let candidates = match arch {
        Architecture::Qwen3 => vec!["qwen_tokenizer.json", "tokenizer.json"],
        Architecture::Llama | Architecture::DeepSeek => vec!["llama_tokenizer.json", "tokenizer.json"],
        Architecture::Gemma => vec!["gemma_tokenizer.json", "tokenizer.json"],
        Architecture::Mistral3 => vec!["mistral_tokenizer.json", "tokenizer.json"],
        _ => vec!["tokenizer.json"],
    };
    for c in candidates {
        if Path::new(c).exists() {
            if let Ok(t) = tokenizers::Tokenizer::from_file(c) {
                return Some(t);
            }
        }
    }

    // 3. Fallback to download from HuggingFace Hub (public un-gated repos)
    let repo_fallbacks: Vec<&str> = match arch {
        Architecture::Qwen3 => vec!["Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-0.5B"],
        Architecture::Llama | Architecture::DeepSeek => vec![
            "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            "NousResearch/Meta-Llama-3-8B-Instruct",
            "meta-llama/Meta-Llama-3-8B-Instruct",
        ],
        Architecture::Gemma => vec!["google/gemma-2-2b-it", "google/gemma-2-9b"],
        Architecture::Mistral3 => vec!["mistralai/Mistral-7B-Instruct-v0.3"],
        _ => return None,
    };

    if let Ok(api) = hf_hub::api::sync::Api::new() {
        for repo_name in repo_fallbacks {
            let repo = api.model(repo_name.to_string());
            if let Ok(tok_file) = repo.get("tokenizer.json") {
                if let Ok(tok) = tokenizers::Tokenizer::from_file(tok_file) {
                    return Some(tok);
                }
            }
        }
    }
    None
}

/// Build a `VarBuilder` exposing a standalone safetensors file's keys verbatim (no remapping).
/// Used to attach the SD3.5 text encoders (CLIP-L / CLIP-G / T5-XXL are separate files).
fn vb_from_file(path: &Path, device: &Device, dtype: DType) -> Result<candle_nn::VarBuilder<'static>> {
    let archive = if path.is_dir() {
        SafeTensorsArchive::open_shards_dir(path)?
    } else {
        SafeTensorsArchive::open(path)?
    };
    let mut map = std::collections::HashMap::new();
    for key in archive.tensor_names() {
        map.insert(key.clone(), archive.get_tensor(&key, device, dtype)?);
    }
    Ok(candle_nn::VarBuilder::from_tensors(map, dtype, device))
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
