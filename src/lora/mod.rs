// [WFGY] Zone: SAFE | Î»: 0.20 | Fallbacks: 0 | Action: LoRA & LyCORIS engine root module

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
    /// Per-loaded-LoRA delta map (one entry per LoRA), so each LoRA's weight can be re-adjusted
    /// independently. `applied_deltas` is the derived element-wise sum.
    per_lora_deltas: Vec<HashMap<String, Tensor>>,
    applied_deltas: HashMap<String, Tensor>,
    device: Device,
    dtype: DType,
}

impl LoRAManager {
    pub fn new() -> Self {
        Self {
            loaded_loras: Vec::new(),
            per_lora_deltas: Vec::new(),
            applied_deltas: HashMap::new(),
            device: Device::Cpu,
            dtype: DType::F32,
        }
    }

    pub fn load_and_merge<P: AsRef<Path>>(
        &mut self,
        path: P,
        multiplier: f64,
        device: &Device,
        dtype: DType,
    ) -> Result<Vec<(String, Tensor)>> {
        self.device = device.clone();
        self.dtype = dtype;
        let loaded = LoRALoader::load_from_file(path, multiplier, device, dtype)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        // Compute this LoRA's deltas (keyed by param), recomputed on weight change.
        let mut this_deltas: HashMap<String, Tensor> = HashMap::new();
        let mut computed_deltas = Vec::new();
        for pair in &loaded.pairs {
            let delta = LoRAMerger::compute_delta(pair, multiplier)?
                .to_device(device)?
                .to_dtype(dtype)?;
            let param_name = pair.target_param.clone();
            if let Some(existing) = this_deltas.get_mut(&param_name) {
                *existing = (existing.as_ref() + &delta)?;
            } else {
                this_deltas.insert(param_name.clone(), delta.clone());
            }
            computed_deltas.push((param_name, delta));
        }

        self.per_lora_deltas.push(this_deltas);
        self.loaded_loras.push(loaded);
        self.recompute_applied()?;
        Ok(computed_deltas)
    }

    /// Re-weight an already-loaded LoRA (by index) without re-loading the file. The multiplier is a
    /// task weight (e.g. 0.20 == 20%); delta contributions are recomputed and the running sum updated.
    pub fn set_multiplier(&mut self, index: usize, multiplier: f64) -> Result<()> {
        if index >= self.loaded_loras.len() {
            return Err(candle_core::Error::Msg(format!(
                "LoRA index {index} out of range (loaded: {})", self.loaded_loras.len()
            )));
        }
        self.loaded_loras[index].multiplier = multiplier;
        let device = self.device.clone();
        let dtype = self.dtype;
        let mut remapped: HashMap<String, Tensor> = HashMap::new();
        for pair in &self.loaded_loras[index].pairs {
            let delta = LoRAMerger::compute_delta(pair, multiplier)?
                .to_device(&device)?
                .to_dtype(dtype)?;
            let param = pair.target_param.clone();
            if let Some(existing) = remapped.get_mut(&param) {
                *existing = (existing.as_ref() + &delta)?;
            } else {
                remapped.insert(param, delta);
            }
        }
        self.per_lora_deltas[index] = remapped;
        self.recompute_applied()?;
        Ok(())
    }

    /// Recompute the summed applied deltas from the per-LoRA contributions.
    fn recompute_applied(&mut self) -> Result<()> {
        let mut sum: HashMap<String, Tensor> = HashMap::new();
        for per in &self.per_lora_deltas {
            for (k, v) in per {
                if let Some(existing) = sum.get_mut(k) {
                    *existing = (existing.as_ref() + v)?;
                } else {
                    sum.insert(k.clone(), v.clone());
                }
            }
        }
        self.applied_deltas = sum;
        Ok(())
    }

    pub fn loaded_loras(&self) -> &[LoadedLoRA] {
        &self.loaded_loras
    }

    pub fn applied_deltas(&self) -> &HashMap<String, Tensor> {
        &self.applied_deltas
    }

    /// The per-LoRA delta map for a single loaded LoRA (by index). Used by in-place (SDXL) pipelines
    /// to subtract the old contribution before applying a re-weighted one.
    pub fn lora_deltas(&self, index: usize) -> Option<&HashMap<String, Tensor>> {
        self.per_lora_deltas.get(index)
    }

    pub fn clear(&mut self) {
        self.loaded_loras.clear();
        self.per_lora_deltas.clear();
        self.applied_deltas.clear();
    }

    pub fn is_empty(&self) -> bool {
        self.loaded_loras.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lora::types::LoRAPair;

    fn pair(name: &str, v: f32) -> LoRAPair {
        let dev = Device::Cpu;
        // down [r, in] = [1,2], up [out, r] = [2,1] -> delta [out,in] = [2,2]
        let down = Tensor::from_vec(vec![v, v], (1, 2), &dev).unwrap();
        let up = Tensor::from_vec(vec![v, v], (2, 1), &dev).unwrap();
        LoRAPair {
            name: name.into(),
            target: LoRATarget::Flux,
            target_param: format!("{name}.weight"),
            down,
            up,
            alpha: Some(1.0),
            rank: 1,
            scale: 1.0,
        }
    }

    fn loaded(path: &str, pairs: Vec<LoRAPair>) -> LoadedLoRA {
        LoadedLoRA { path: path.into(), multiplier: 1.0, pairs }
    }

    #[test]
    fn set_multiplier_reweights_and_sum() {
        let dev = Device::Cpu;
        let mut mgr = LoRAManager::new();
        mgr.device = dev.clone();
        mgr.dtype = DType::F32;

        // A 2x2 delta of all-ones scaled by weight w: (up@down) with unit vectors = [[1,1],[1,1]],
        // so its abs-sum = 4. Compute_delta multiplies by eff_scale = mult * scale (=mult since scale=1).
        let l1 = loaded("a", vec![pair("p", 1.0)]);
        let l2 = loaded("b", vec![pair("p", 1.0)]);
        let mut d1 = HashMap::new();
        d1.insert("p.weight".to_string(), LoRAMerger::compute_delta(&l1.pairs[0], 0.2).unwrap());
        mgr.per_lora_deltas.push(d1);
        mgr.loaded_loras.push(l1);
        let mut d2 = HashMap::new();
        d2.insert("p.weight".to_string(), LoRAMerger::compute_delta(&l2.pairs[0], 0.4).unwrap());
        mgr.per_lora_deltas.push(d2);
        mgr.loaded_loras.push(l2);
        mgr.recompute_applied().unwrap();

        let abs_sum = |mgr: &LoRAManager| -> f32 { mgr.applied_deltas["p.weight"].flatten_all().unwrap().to_vec1::<f32>().unwrap().iter().map(|x| x.abs()).sum() };

        // 0.2*[1s] + 0.4*[1s] -> abs-sum = 4 * 0.6 = 2.4
        let before = abs_sum(&mgr);
        assert!((before - 2.4).abs() < 1e-3, "before={before}");

        // Re-weight lora0 to 0.5 -> (0.5+0.4)*[1s] -> abs-sum = 4 * 0.9 = 3.6
        mgr.set_multiplier(0, 0.5).unwrap();
        let after = abs_sum(&mgr);
        assert!((after - 3.6).abs() < 1e-3, "after={after}");
        assert_eq!(mgr.loaded_loras()[0].multiplier, 0.5);
    }

    #[test]
    fn multiple_loras_accumulate() {
        let dev = Device::Cpu;
        let mut mgr = LoRAManager::new();
        mgr.device = dev.clone();
        mgr.dtype = DType::F32;

        let l1 = loaded("a", vec![pair("p", 1.0)]);
        let l2 = loaded("b", vec![pair("q", 1.0)]);
        let mut d1 = HashMap::new();
        d1.insert("p.weight".to_string(), LoRAMerger::compute_delta(&l1.pairs[0], 1.0).unwrap());
        mgr.per_lora_deltas.push(d1);
        mgr.loaded_loras.push(l1);
        let mut d2 = HashMap::new();
        d2.insert("q.weight".to_string(), LoRAMerger::compute_delta(&l2.pairs[0], 1.0).unwrap());
        mgr.per_lora_deltas.push(d2);
        mgr.loaded_loras.push(l2);
        mgr.recompute_applied().unwrap();

        assert!(mgr.applied_deltas.contains_key("p.weight"));
        assert!(mgr.applied_deltas.contains_key("q.weight"));
        assert_eq!(mgr.applied_deltas.len(), 2);
    }
}

