// [WFGY] Zone: SAFE | λ: 0.30 | Fallbacks: 0 | Action: Unified model resolver — turn a model origin (local disk path, HF repo/mirror) into local weight paths

use crate::error::{LuminaError, Result};
use hf_hub::api::sync::{Api, ApiBuilder};
use hf_hub::{Repo, RepoType};
use std::path::{Path, PathBuf};

/// Where a model's weights come from, for building with the community's common origins.
///
/// - **Local** — a model already on disk (the Civitai workflow: the user points at a file/folder).
/// - **Hf** — a Hugging Face (or mirror such as Citai) repo id plus the files inside it.
#[derive(Clone, Debug)]
pub enum ModelOrigin {
    /// A path already on disk: a weight/binding folder or a single `.safetensors` file.
    Local(PathBuf),
    /// A Hugging Face (or mirror) repo `repo` plus one or more file names inside it.
    /// `None` revision means the repo's default branch.
    Hf {
        repo: String,
        files: Vec<String>,
        revision: Option<String>,
    },
}

/// Central brick that resolves a [`ModelOrigin`] into a set of local file paths, downloading the
/// HF/mirror ones when needed. It is built on `hf_hub`'s environment-aware builder so the HF
/// endpoint and cache are configurable externally rather than hard-coded:
/// - `HF_ENDPOINT` — point at any Hugging Face-compatible mirror (e.g. Citai) with **no code change**.
/// - `HF_HOME` / `HF_HUB_CACHE` — override where resolved HF files are cached.
/// - `HF_TOKEN` — authenticate for gated/private repos.
///
/// Downstream bricks (text encoders, DiT, VAE) receive a `&ModelHub` and call
/// [`ModelHub::resolve`] / [`ModelHub::resolve_hf`], assembling interchangeable parts without
/// per-format or per-origin URLs. Local (Civitai) models are addressed purely by path.
#[derive(Clone)]
pub struct ModelHub {
    api: Api,
    cache_dir: PathBuf,
}

impl ModelHub {
    /// Open the hub using environment defaults (`HF_ENDPOINT`, `HF_HOME`, `HF_TOKEN`).
    pub fn from_env() -> Result<Self> {
        let cache_dir = hf_cache_dir();
        let api = ApiBuilder::from_env().build().map_err(hub_err)?;
        Ok(Self { api, cache_dir })
    }

    /// Open the hub targeting a specific cache directory. Other env controls still apply.
    pub fn with_cache_dir(cache_dir: impl AsRef<Path>) -> Result<Self> {
        let cache_dir = cache_dir.as_ref().to_path_buf();
        let api = ApiBuilder::from_env()
            .with_cache_dir(cache_dir.clone())
            .build()
            .map_err(hub_err)?;
        Ok(Self { api, cache_dir })
    }

    /// Open the hub against an explicit endpoint (e.g. a Citai mirror), overriding `HF_ENDPOINT`.
    pub fn with_endpoint(endpoint: impl Into<String>, cache_dir: impl AsRef<Path>) -> Result<Self> {
        let cache_dir = cache_dir.as_ref().to_path_buf();
        let api = ApiBuilder::from_env()
            .with_endpoint(endpoint.into())
            .with_cache_dir(cache_dir.clone())
            .build()
            .map_err(hub_err)?;
        Ok(Self { api, cache_dir })
    }

    /// Ensure a single file of a HF (or mirror) repo is available locally and return its path.
    pub fn resolve_hf(&self, repo: &str, filename: &str, revision: Option<&str>) -> Result<PathBuf> {
        let repo = match revision {
            Some(rev) => Repo::with_revision(repo.to_string(), RepoType::Model, rev.to_string()),
            None => Repo::model(repo.to_string()),
        };
        self.api.repo(repo).get(filename).map_err(hub_err)
    }

    /// Ensure every file of a HF (or mirror) repo is available, returning the local folder.
    /// Useful for multi-shard checkpoints (`model-00001-of-00005.safetensors`, ...).
    pub fn resolve_hf_repo(&self, repo: &str, files: &[String], revision: Option<&str>) -> Result<PathBuf> {
        let mut dir = None;
        for f in files {
            let p = self.resolve_hf(repo, f, revision)?;
            if dir.is_none() {
                dir = p.parent().map(|d| d.to_path_buf());
            }
        }
        dir.ok_or_else(|| LuminaError::Context {
            context: format!("no files to resolve for repo {repo}"),
            source: Box::new(LuminaError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                repo.to_string(),
            ))),
        })
    }

    /// General resolver. For a [`ModelOrigin::Local`] it returns the given path (folder or parent of a
    /// file); for [`ModelOrigin::Hf`] it ensures the files exist and returns their holding folder.
    pub fn resolve(&self, origin: &ModelOrigin) -> Result<PathBuf> {
        match origin {
            ModelOrigin::Local(p) => {
                if p.is_dir() {
                    Ok(p.clone())
                } else if p.is_file() {
                    Ok(p.parent().unwrap_or(p).to_path_buf())
                } else {
                    Err(LuminaError::Context {
                        context: format!("local model path not found: {}", p.display()),
                        source: Box::new(LuminaError::Io(std::io::Error::new(
                            std::io::ErrorKind::NotFound,
                            "",
                        ))),
                    })
                }
            }
            ModelOrigin::Hf { repo, files, revision } => {
                self.resolve_hf_repo(repo, files, revision.as_deref())
            }
        }
    }

    /// The configured output/cache directory, for display / binding to a downloader.
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }
}

impl std::fmt::Debug for ModelHub {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelHub")
            .field("cache_dir", &self.cache_dir)
            .finish_non_exhaustive()
    }
}

fn hub_err<T>(e: T) -> LuminaError
where
    T: std::fmt::Display,
{
    LuminaError::Context {
        context: format!("model hub error: {e}"),
        source: Box::new(LuminaError::Io(std::io::Error::new(
            std::io::ErrorKind::Other,
            e.to_string(),
        ))),
    }
}

/// Best-effort default HF cache dir, mirroring `~/.cache/huggingface` (respecting `HF_HOME`).
fn hf_cache_dir() -> PathBuf {
    if let Ok(home) = std::env::var("HF_HOME") {
        return PathBuf::from(home);
    }
    match std::env::var("HF_HUB_CACHE") {
        Ok(cache) => PathBuf::from(cache),
        Err(_) => {
            let home = std::env::var("HOME")
                .or_else(|_| std::env::var("USERPROFILE"))
                .unwrap_or_default();
            PathBuf::from(home).join(".cache").join("huggingface")
        }
    }
}
