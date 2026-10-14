// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: Cross-Modal Tensor Routing & Shared Pipeline VRAM Cache

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use candle_core::Tensor;
use crate::error::{LuminaError, Result};

/// Modality partition for pipeline execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PipelineModality {
    Vision,
    VisionLanguage,
    Text,
    Audio,
}

/// Cache entry tracking tensor buffer residency and allocation overhead.
#[derive(Clone)]
struct CacheEntry {
    tensor: Tensor,
    size_bytes: usize,
    modality: PipelineModality,
    last_access_tick: u64,
}

/// Shared cross-modal pipeline cache managing VRAM memory budgets and deterministic LRU eviction
/// across Vision (Flux / SDXL), Multimodal (Qwen-VL / PaliGemma), Text (Qwen / Gemma / LLaMA), and Audio (Whisper / TTS).
pub struct SharedPipelineCache {
    capacity_bytes: usize,
    used_bytes: usize,
    entries: HashMap<String, CacheEntry>,
    current_tick: u64,
}

impl SharedPipelineCache {
    /// Create a cache pool with a given maximum capacity in bytes (e.g. 12 * 1024 * 1024 * 1024 for 12 GB VRAM).
    pub fn new(capacity_bytes: usize) -> Self {
        Self {
            capacity_bytes,
            used_bytes: 0,
            entries: HashMap::new(),
            current_tick: 0,
        }
    }

    /// Default 12 GB VRAM allocation pool.
    pub fn default_12gb() -> Self {
        Self::new(12 * 1024 * 1024 * 1024)
    }

    /// Insert or update a resident tensor buffer. If capacity is exceeded, triggers deterministic LRU eviction.
    pub fn insert(
        &mut self,
        key: impl Into<String>,
        modality: PipelineModality,
        tensor: Tensor,
    ) -> Result<()> {
        let key = key.into();
        self.current_tick += 1;

        let num_elements = tensor.elem_count();
        let bytes_per_elem = tensor.dtype().size_in_bytes();
        let size_bytes = num_elements * bytes_per_elem;

        // If replacing an existing key, subtract its previous size
        if let Some(old) = self.entries.remove(&key) {
            self.used_bytes = self.used_bytes.saturating_sub(old.size_bytes);
        }

        // Evict LRU entries if adding this tensor exceeds capacity
        if self.used_bytes + size_bytes > self.capacity_bytes {
            self.evict_lru(size_bytes)?;
        }

        self.used_bytes += size_bytes;
        self.entries.insert(
            key,
            CacheEntry {
                tensor,
                size_bytes,
                modality,
                last_access_tick: self.current_tick,
            },
        );

        Ok(())
    }

    /// Retrieve a cached tensor and refresh its access tick.
    pub fn get(&mut self, key: &str) -> Option<Tensor> {
        self.current_tick += 1;
        let tick = self.current_tick;
        if let Some(entry) = self.entries.get_mut(key) {
            entry.last_access_tick = tick;
            Some(entry.tensor.clone())
        } else {
            None
        }
    }

    /// Evict all resident tensors belonging to a specific modality (e.g. freeing Vision VRAM when switching to Text).
    pub fn evict_modality(&mut self, modality: PipelineModality) -> usize {
        let mut freed = 0;
        let to_remove: Vec<String> = self
            .entries
            .iter()
            .filter(|(_, e)| e.modality == modality)
            .map(|(k, _)| k.clone())
            .collect();

        for k in to_remove {
            if let Some(entry) = self.entries.remove(&k) {
                freed += entry.size_bytes;
                self.used_bytes = self.used_bytes.saturating_sub(entry.size_bytes);
            }
        }
        freed
    }

    /// Deterministically evict the least-recently-used tensors until sufficient free bytes exist.
    pub fn evict_lru(&mut self, required_bytes: usize) -> Result<()> {
        while self.used_bytes + required_bytes > self.capacity_bytes && !self.entries.is_empty() {
            // Find lowest last_access_tick
            let oldest_key = self
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.last_access_tick)
                .map(|(k, _)| k.clone());

            if let Some(k) = oldest_key {
                if let Some(removed) = self.entries.remove(&k) {
                    self.used_bytes = self.used_bytes.saturating_sub(removed.size_bytes);
                }
            } else {
                break;
            }
        }

        if self.used_bytes + required_bytes > self.capacity_bytes {
            return Err(LuminaError::Config(format!(
                "Required {} bytes exceeds maximum pool capacity {} bytes (used: {})",
                required_bytes, self.capacity_bytes, self.used_bytes
            )));
        }

        Ok(())
    }

    /// Current bytes allocated in the pool.
    pub fn used_bytes(&self) -> usize {
        self.used_bytes
    }

    /// Pool maximum budget in bytes.
    pub fn capacity_bytes(&self) -> usize {
        self.capacity_bytes
    }

    /// Number of resident buffers.
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }
}

/// Global thread-safe shared pipeline cache singleton handle.
#[derive(Clone)]
pub struct SharedPipelineCacheHandle {
    inner: Arc<Mutex<SharedPipelineCache>>,
}

impl SharedPipelineCacheHandle {
    pub fn new(capacity_bytes: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(SharedPipelineCache::new(capacity_bytes))),
        }
    }

    pub fn lock(&self) -> std::sync::MutexGuard<'_, SharedPipelineCache> {
        self.inner.lock().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_shared_cache_lru_and_modality_eviction() {
        // Pool capacity = 1000 bytes
        let mut pool = SharedPipelineCache::new(1000);

        // Tensor 1: 50 f32 = 200 bytes (Vision)
        let t1 = Tensor::zeros((50,), candle_core::DType::F32, &Device::Cpu).unwrap();
        pool.insert("t1", PipelineModality::Vision, t1).unwrap();
        assert_eq!(pool.used_bytes(), 200);

        // Tensor 2: 100 f32 = 400 bytes (Text)
        let t2 = Tensor::zeros((100,), candle_core::DType::F32, &Device::Cpu).unwrap();
        pool.insert("t2", PipelineModality::Text, t2).unwrap();
        assert_eq!(pool.used_bytes(), 600);

        // Tensor 3: 125 f32 = 500 bytes (VisionLanguage) -> exceeds 1000 bytes (600 + 500 = 1100 > 1000)
        // t1 is oldest -> t1 evicted, pool has t2 (400) + t3 (500) = 900 bytes
        let t3 = Tensor::zeros((125,), candle_core::DType::F32, &Device::Cpu).unwrap();
        pool.insert("t3", PipelineModality::VisionLanguage, t3).unwrap();
        assert_eq!(pool.used_bytes(), 900);
        assert!(pool.get("t1").is_none());
        assert!(pool.get("t2").is_some());
        assert!(pool.get("t3").is_some());

        // Evict text modality -> t2 removed (400 bytes freed), leaving t3 (500 bytes)
        let freed = pool.evict_modality(PipelineModality::Text);
        assert_eq!(freed, 400);
        assert_eq!(pool.used_bytes(), 500);
        assert!(pool.get("t2").is_none());
        assert!(pool.get("t3").is_some());
    }
}

