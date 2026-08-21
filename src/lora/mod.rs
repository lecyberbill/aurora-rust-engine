// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: LoRA & LyCORIS engine root module

pub mod loader;
pub mod merger;
pub mod types;

pub use loader::LoRALoader;
pub use merger::LoRAMerger;
pub use types::{LoRAPair, LoRATarget, LoadedLoRA};

use std::collections::HashMap;
use std::path::Path;
use candle_core::{DType, Device, Result, Tensor};

pub struct LoRAManager {
    loaded_loras: Vec<LoadedLoRA>,
    applied_deltas: HashMap<String, Tensor>,
}

impl LoRAManager {
    pub fn new() -> Self {
        Self {
            loaded_loras: Vec::new(),
            applied_deltas: HashMap::new(),
        }
    }

    pub fn load_and_merge<P: AsRef<Path>>(
        &mut self,
        path: P,
        multiplier: f64,
        device: &Device,
        dtype: DType,
    ) -> Result<Vec<(String, Tensor)>> {
        let loaded = LoRALoader::load_from_file(path, multiplier, device, dtype)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        let mut computed_deltas = Vec::new();

        for pair in &loaded.pairs {
            let delta = LoRAMerger::compute_delta(pair, multiplier)?
                .to_device(device)?
                .to_dtype(dtype)?;
            let param_name = pair.target_param.clone();

            if let Some(existing) = self.applied_deltas.get_mut(&param_name) {
                *existing = (existing.as_ref() + &delta)?;
            } else {
                self.applied_deltas.insert(param_name.clone(), delta.clone());
            }

            computed_deltas.push((param_name, delta));
        }

        self.loaded_loras.push(loaded);
        Ok(computed_deltas)
    }

    pub fn loaded_loras(&self) -> &[LoadedLoRA] {
        &self.loaded_loras
    }

    pub fn applied_deltas(&self) -> &HashMap<String, Tensor> {
        &self.applied_deltas
    }

    pub fn clear(&mut self) {
        self.loaded_loras.clear();
        self.applied_deltas.clear();
    }

    pub fn is_empty(&self) -> bool {
        self.loaded_loras.is_empty()
    }
}
