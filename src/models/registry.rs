// [WFGY] Zone: SAFE | λ: 0.25 | Fallbacks: 0 | Action: In-process model cache + registry (load / get / unload / swap)

use candle_core::{DType, Device};
use std::collections::HashMap;
use std::sync::RwLock;
use crate::error::Result;
use crate::hub::ModelHub;
use super::auto::{AutoModel, BoxedModel, ModelLoadConfig};
use super::common::ModelKind;

/// Metadata about a registered model (no weights — cheap to list).
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub id: String,
    pub repo: String,
    pub kind: ModelKind,
    pub family: String,
    pub device: Device,
    pub dtype: DType,
}

/// In-process registry of loaded models. Concurrency-safe (read-mostly cache behind `RwLock`).
/// It owns the [`ModelHub`] so `load`/`swap` can materialise checkpoints on demand via
/// [`AutoModel::from_pretrained`]. This is the platform-facing handle: an HTTP backend or a server
/// crate holds one `ModelRegistry` and serves every model from a single `id -> model` map.
pub struct ModelRegistry {
    hub: ModelHub,
    device: Device,
    dtype: DType,
    models: RwLock<HashMap<String, BoxedModel>>,
}

impl ModelRegistry {
    /// Build a registry that resolves HF/mirror checkpoints through `hub` and materialises models on
    /// the given device/dtype.
    pub fn new(hub: ModelHub, device: Device, dtype: DType) -> Self {
        Self { hub, device, dtype, models: RwLock::new(HashMap::new()) }
    }

    /// Build with environment-default hub and an explicit device/dtype.
    pub fn new_with_default_hub(device: Device, dtype: DType) -> Self {
        let hub = ModelHub::from_env().unwrap_or_else(|_| ModelHub::with_cache_dir(std::env::temp_dir()).unwrap());
        Self::new(hub, device, dtype)
    }

    pub fn hub(&self) -> &ModelHub { &self.hub }

    /// Load (and cache) a model by its `repo` id. Returns the model handle; a second call with the
    /// same id is a cache hit.
    pub fn load(&self, model_id: &str, repo: &str) -> Result<BoxedModel> {
        if let Some(m) = self.get(model_id) {
            return Ok(m);
        }
        let cfg = ModelLoadConfig::new(repo, self.device.clone(), self.dtype);
        let model = AutoModel::from_pretrained(&self.hub, cfg)?;
        self.insert(model_id.to_string(), model.clone());
        Ok(model)
    }

    /// Register a model already built by the caller (e.g. from a local path).
    pub fn insert(&self, model_id: String, model: BoxedModel) -> BoxedModel {
        let m = model.clone();
        self.models.write().unwrap().insert(model_id, model);
        m
    }

    pub fn get(&self, model_id: &str) -> Option<BoxedModel> {
        self.models.read().unwrap().get(model_id).cloned()
    }

    /// Reload a model under the same id with a new repo/checkpoint (hot swap).
    pub fn swap(&self, model_id: &str, repo: &str) -> Result<BoxedModel> {
        let cfg = ModelLoadConfig::new(repo, self.device.clone(), self.dtype);
        let model = AutoModel::from_pretrained(&self.hub, cfg)?;
        self.models.write().unwrap().insert(model_id.to_string(), model.clone());
        Ok(model)
    }

    /// Unload a model, returning whether it was present.
    pub fn unload(&self, model_id: &str) -> bool {
        self.models.write().unwrap().remove(model_id).is_some()
    }

    /// Remove everything (e.g. on shutdown, to free VRAM).
    pub fn clear(&self) {
        self.models.write().unwrap().clear();
    }

    /// Cheap metadata snapshot of every registered model.
    pub fn list(&self) -> Vec<ModelInfo> {
        let guard = self.models.read().unwrap();
        let mut out = Vec::with_capacity(guard.len());
        for (id, m) in guard.iter() {
            let inner = m.lock().map_err(|_| ()).ok();
            let (kind, family, dev, dt) = inner
                .map(|g| (g.kind(), g.family().to_string(), g.device().clone(), g.dtype()))
                .unwrap_or_else(|| (ModelKind::Unknown, String::new(), self.device.clone(), self.dtype));
            out.push(ModelInfo { id: id.clone(), repo: String::new(), kind, family, device: dev, dtype: dt });
        }
        out
    }

    pub fn len(&self) -> usize {
        self.models.read().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.models.read().unwrap().is_empty()
    }
}
