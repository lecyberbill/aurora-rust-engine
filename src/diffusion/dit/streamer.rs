// [WFGY] Zone: SAFE | λ: 0.25 | Fallbacks: 0 | Action: Sequential Block Streamer for MMDiT Low-VRAM Inference

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use std::collections::HashMap;
use std::sync::Arc;
use crate::diffusion::dit::blocks::{DoubleStreamBlock, SingleStreamBlock};
use crate::weights::{SafeTensorsArchive, apply_flux_deltas_to_tensor};

/// Stream-loads individual MMDiT blocks into GPU VRAM on-demand and drops them after computation
pub struct SequentialBlockStreamer {
    archive: Arc<SafeTensorsArchive>,
    device: Device,
    dtype: DType,
    hidden_dim: usize,
    num_heads: usize,
    mlp_ratio: usize,
    /// Optional LoRA deltas (BFL-style names, possibly `@Q`/`@K`/`@V`-tagged) to splice into each
    /// block's weights as it is streamed in.
    lora_deltas: Option<Arc<HashMap<String, Tensor>>>,
}

impl SequentialBlockStreamer {
    pub fn new(
        archive: Arc<SafeTensorsArchive>,
        device: Device,
        dtype: DType,
        hidden_dim: usize,
        num_heads: usize,
        mlp_ratio: usize,
    ) -> Self {
        Self {
            archive,
            device,
            dtype,
            hidden_dim,
            num_heads,
            mlp_ratio,
            lora_deltas: None,
        }
    }

    /// Attach LoRA deltas (BFL-style names, possibly `@Q`/`@K`/`@V`-tagged) to splice into each
    /// streamed block's weights.
    pub fn set_lora_deltas(&mut self, lora_deltas: HashMap<String, Tensor>) {
        self.lora_deltas = Some(Arc::new(lora_deltas));
    }

    /// Clear any attached LoRA deltas.
    pub fn clear_lora_deltas(&mut self) {
        self.lora_deltas = None;
    }

    /// Load and execute a single DoubleStreamBlock on GPU, then return result
    pub fn execute_double_block(
        &self,
        block_idx: usize,
        img: &Tensor,
        txt: &Tensor,
        temb: &Tensor,
        img_freqs_cos: Option<&Tensor>,
        img_freqs_sin: Option<&Tensor>,
        txt_freqs_cos: Option<&Tensor>,
        txt_freqs_sin: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let prefix = format!("double_blocks.{}.", block_idx);
        let prefix_alt = format!("model.diffusion_model.double_blocks.{}.", block_idx);
        let mut tensors = HashMap::new();

        for key in self.archive.keys() {
            let matched_suffix = if let Some(suffix) = key.strip_prefix(&prefix) {
                Some(suffix)
            } else if let Some(suffix) = key.strip_prefix(&prefix_alt) {
                Some(suffix)
            } else {
                None
            };

            if let Some(suffix) = matched_suffix {
                let t = self.archive.get_tensor(key, &self.device, self.dtype)
                    .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                let t = if let Some(deltas) = &self.lora_deltas {
                    apply_flux_deltas_to_tensor(deltas, &format!("{prefix}{suffix}"), t, &self.device, self.dtype)
                        .map_err(|e| candle_core::Error::Msg(e.to_string()))?
                } else { t };
                tensors.insert(suffix.to_string(), t);
            }
        }

        // Inject shared global modulations if block-local ones are absent (Klein architecture)
        let double_mod_dim = self.hidden_dim * 6;
        if !tensors.keys().any(|k| k.starts_with("img_mod")) {
            let t_opt = self.archive.get_tensor("double_stream_modulation_img.lin.weight", &self.device, self.dtype)
                .or_else(|_| self.archive.get_tensor("model.diffusion_model.double_stream_modulation_img.lin.weight", &self.device, self.dtype));
            if let Ok(t) = t_opt {
                let t_slice = if t.dim(0)? == double_mod_dim {
                    t
                } else if t.dim(0)? > double_mod_dim {
                    t.narrow(0, block_idx * double_mod_dim, double_mod_dim)?
                } else {
                    t
                };
                tensors.insert("img_mod.lin.weight".to_string(), t_slice);
            }
        }
        if !tensors.keys().any(|k| k.starts_with("txt_mod")) {
            let t_opt = self.archive.get_tensor("double_stream_modulation_txt.lin.weight", &self.device, self.dtype)
                .or_else(|_| self.archive.get_tensor("model.diffusion_model.double_stream_modulation_txt.lin.weight", &self.device, self.dtype));
            if let Ok(t) = t_opt {
                let t_slice = if t.dim(0)? == double_mod_dim {
                    t
                } else if t.dim(0)? > double_mod_dim {
                    t.narrow(0, block_idx * double_mod_dim, double_mod_dim)?
                } else {
                    t
                };
                tensors.insert("txt_mod.lin.weight".to_string(), t_slice);
            }
        }

        let vb = VarBuilder::from_tensors(tensors, self.dtype, &self.device);
        let block = DoubleStreamBlock::new(self.hidden_dim, self.num_heads, self.mlp_ratio, vb)?;
        block.forward(img, txt, temb, img_freqs_cos, img_freqs_sin, txt_freqs_cos, txt_freqs_sin)
    }

    /// Load and execute a single SingleStreamBlock on GPU, then return result
    pub fn execute_single_block(
        &self,
        block_idx: usize,
        x: &Tensor,
        temb: &Tensor,
        freqs_cos: Option<&Tensor>,
        freqs_sin: Option<&Tensor>,
    ) -> Result<Tensor> {
        let prefix = format!("single_blocks.{}.", block_idx);
        let prefix_alt = format!("model.diffusion_model.single_blocks.{}.", block_idx);
        let mut tensors = HashMap::new();

        for key in self.archive.keys() {
            let matched_suffix = if let Some(suffix) = key.strip_prefix(&prefix) {
                Some(suffix)
            } else if let Some(suffix) = key.strip_prefix(&prefix_alt) {
                Some(suffix)
            } else {
                None
            };

            if let Some(suffix) = matched_suffix {
                let t = self.archive.get_tensor(key, &self.device, self.dtype)
                    .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                let t = if let Some(deltas) = &self.lora_deltas {
                    apply_flux_deltas_to_tensor(deltas, &format!("{prefix}{suffix}"), t, &self.device, self.dtype)
                        .map_err(|e| candle_core::Error::Msg(e.to_string()))?
                } else { t };
                tensors.insert(suffix.to_string(), t);
            }
        }

        // Inject shared single modulation if block-local modulation is absent (Klein architecture)
        let single_mod_dim = self.hidden_dim * 3;
        if !tensors.keys().any(|k| k.starts_with("modulation")) {
            let t_opt = self.archive.get_tensor("single_stream_modulation.lin.weight", &self.device, self.dtype)
                .or_else(|_| self.archive.get_tensor("model.diffusion_model.single_stream_modulation.lin.weight", &self.device, self.dtype));
            if let Ok(t) = t_opt {
                let t_slice = if t.dim(0)? == single_mod_dim {
                    t
                } else if t.dim(0)? > single_mod_dim {
                    t.narrow(0, block_idx * single_mod_dim, single_mod_dim)?
                } else {
                    t
                };
                tensors.insert("modulation.lin.weight".to_string(), t_slice);
            }
        }

        let vb = VarBuilder::from_tensors(tensors, self.dtype, &self.device);
        let block = SingleStreamBlock::new(self.hidden_dim, self.num_heads, self.mlp_ratio, vb)?;
        block.forward(x, temb, freqs_cos, freqs_sin)
    }
}
