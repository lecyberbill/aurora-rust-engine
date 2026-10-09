// [WFGY] Zone: SAFE | λ: 0.25 | Fallbacks: 0 | Action: High-Performance Multiplatform Block Streamer with Double-Buffering & VRAM Weight Caching
// Invariant: Pure multiplatform Rust (CUDA, ROCm, Metal, CPU) with zero data races, bounded VRAM and exact numerical determinism.

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use std::collections::HashMap;
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use crate::canonical::{canonicalize, detect_family, CheckpointFamily};
use crate::diffusion::dit::blocks::{DoubleStreamBlock, SingleStreamBlock};
use crate::weights::{WeightsSource, apply_flux_deltas_to_tensor};

/// Stream-loads individual MMDiT blocks into GPU VRAM on-demand, with optional LRU/VRAM caching
/// and asynchronous double-buffering.
/// Works with any [`WeightsSource`] (safetensors single/multi-shard, or GGUF).
pub struct SequentialBlockStreamer {
    archive: Arc<dyn WeightsSource>,
    device: Device,
    dtype: DType,
    hidden_dim: usize,
    num_heads: usize,
    mlp_ratio: usize,
    /// The checkpoint's key convention, detected once. Drives [`canonicalize`].
    family: CheckpointFamily,
    /// Optional LoRA deltas (BFL-style names, possibly `@Q`/`@K`/`@V`-tagged) to splice into each
    /// block's weights as it is streamed in.
    lora_deltas: Option<Arc<HashMap<String, Tensor>>>,
    /// VRAM Block Weight Cache for zero-I/O repeat steps when VRAM permits
    double_blocks_cache: Arc<Mutex<HashMap<usize, HashMap<String, Tensor>>>>,
    single_blocks_cache: Arc<Mutex<HashMap<usize, HashMap<String, Tensor>>>>,
    max_cached_double_blocks: usize,
    max_cached_single_blocks: usize,
}

impl SequentialBlockStreamer {
    pub fn new(
        archive: Arc<dyn WeightsSource>,
        device: Device,
        dtype: DType,
        hidden_dim: usize,
        num_heads: usize,
        mlp_ratio: usize,
    ) -> Self {
        let family = detect_family(archive.keys().iter().map(|s| s.as_str()));
        Self {
            archive,
            device,
            dtype,
            hidden_dim,
            num_heads,
            mlp_ratio,
            family,
            lora_deltas: None,
            double_blocks_cache: Arc::new(Mutex::new(HashMap::new())),
            single_blocks_cache: Arc::new(Mutex::new(HashMap::new())),
            max_cached_double_blocks: 0,
            max_cached_single_blocks: 0,
        }
    }

    /// Enable VRAM Weight Caching (keeps up to `max_double` double blocks and `max_single` single blocks resident in VRAM)
    pub fn with_vram_cache(mut self, max_double: usize, max_single: usize) -> Self {
        self.max_cached_double_blocks = max_double;
        self.max_cached_single_blocks = max_single;
        self
    }

    /// Set VRAM cache capacities dynamically
    pub fn set_cache_capacities(&mut self, max_double: usize, max_single: usize) {
        self.max_cached_double_blocks = max_double;
        self.max_cached_single_blocks = max_single;
    }

    /// Clear all resident cached blocks
    pub fn clear_cache(&self) {
        if let Ok(mut c) = self.double_blocks_cache.lock() {
            c.clear();
        }
        if let Ok(mut c) = self.single_blocks_cache.lock() {
            c.clear();
        }
    }

    /// Attach LoRA deltas (BFL-style names, possibly `@Q`/`@K`/`@V`-tagged) to splice into each
    /// streamed block's weights.
    pub fn set_lora_deltas(&mut self, lora_deltas: HashMap<String, Tensor>) {
        self.clear_cache();
        self.lora_deltas = Some(Arc::new(lora_deltas));
    }

    /// Clear any attached LoRA deltas.
    pub fn clear_lora_deltas(&mut self) {
        self.clear_cache();
        self.lora_deltas = None;
    }

    /// Load block tensors dictionary for DoubleStreamBlock at `block_idx` (with VRAM Cache lookup)
    pub fn load_double_block_tensors(&self, block_idx: usize) -> Result<HashMap<String, Tensor>> {
        if self.max_cached_double_blocks > 0 {
            if let Ok(cache) = self.double_blocks_cache.lock() {
                if let Some(cached_tensors) = cache.get(&block_idx) {
                    return Ok(cached_tensors.clone());
                }
            }
        }

        let prefix = format!("double_blocks.{}.", block_idx);
        let prefix_alt = format!("model.diffusion_model.double_blocks.{}.", block_idx);
        let mut tensors = HashMap::new();

        for key in self.archive.keys() {
            let bfl = canonicalize(self.family, &key).unwrap_or_else(|| key.clone());
            let matched_suffix = if let Some(suffix) = bfl.strip_prefix(&prefix) {
                Some(suffix.to_string())
            } else if let Some(suffix) = bfl.strip_prefix(&prefix_alt) {
                Some(suffix.to_string())
            } else {
                None
            };

            if let Some(suffix) = matched_suffix {
                let t = self.archive.get_tensor(&key, &self.device, self.dtype)
                    .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                let t = if let Some(deltas) = &self.lora_deltas {
                    apply_flux_deltas_to_tensor(deltas, &format!("{prefix}{suffix}"), t, &self.device, self.dtype)
                        .map_err(|e| candle_core::Error::Msg(e.to_string()))?
                } else { t };
                tensors.insert(suffix, t);
            }
        }

        // Fuse Diffusers-layout split QKV (img_attn.qkv@Q/@K/@V) back into a single fused weight.
        fuse_split_qkv(&mut tensors, "img_attn.qkv");
        fuse_split_qkv(&mut tensors, "txt_attn.qkv");

        // Inject shared global modulations if block-local ones are absent (Klein architecture)
        let double_mod_dim = self.hidden_dim * 6;
        if !tensors.keys().any(|k| k.starts_with("img_mod")) {
            let t_opt = self.archive.get_tensor("double_stream_modulation_img.lin.weight", &self.device, self.dtype)
                .or_else(|_| self.archive.get_tensor("double_stream_modulation_img.linear.weight", &self.device, self.dtype))
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
                .or_else(|_| self.archive.get_tensor("double_stream_modulation_txt.linear.weight", &self.device, self.dtype))
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

        if self.max_cached_double_blocks > 0 && block_idx < self.max_cached_double_blocks {
            if let Ok(mut cache) = self.double_blocks_cache.lock() {
                cache.insert(block_idx, tensors.clone());
            }
        }

        Ok(tensors)
    }

    /// Load block tensors dictionary for SingleStreamBlock at `block_idx` (with VRAM Cache lookup)
    pub fn load_single_block_tensors(&self, block_idx: usize) -> Result<HashMap<String, Tensor>> {
        if self.max_cached_single_blocks > 0 {
            if let Ok(cache) = self.single_blocks_cache.lock() {
                if let Some(cached_tensors) = cache.get(&block_idx) {
                    return Ok(cached_tensors.clone());
                }
            }
        }

        let prefix = format!("single_blocks.{}.", block_idx);
        let prefix_alt = format!("model.diffusion_model.single_blocks.{}.", block_idx);
        let mut tensors = HashMap::new();

        for key in self.archive.keys() {
            let bfl = canonicalize(self.family, &key).unwrap_or_else(|| key.clone());
            let matched_suffix = if let Some(suffix) = bfl.strip_prefix(&prefix) {
                Some(suffix.to_string())
            } else if let Some(suffix) = bfl.strip_prefix(&prefix_alt) {
                Some(suffix.to_string())
            } else {
                None
            };

            if let Some(suffix) = matched_suffix {
                let t = self.archive.get_tensor(&key, &self.device, self.dtype)
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
                .or_else(|_| self.archive.get_tensor("single_stream_modulation.linear.weight", &self.device, self.dtype))
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

        if self.max_cached_single_blocks > 0 && block_idx < self.max_cached_single_blocks {
            if let Ok(mut cache) = self.single_blocks_cache.lock() {
                cache.insert(block_idx, tensors.clone());
            }
        }

        Ok(tensors)
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
        let tensors = self.load_double_block_tensors(block_idx)?;
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
        let tensors = self.load_single_block_tensors(block_idx)?;
        let vb = VarBuilder::from_tensors(tensors, self.dtype, &self.device);
        let block = SingleStreamBlock::new(self.hidden_dim, self.num_heads, self.mlp_ratio, vb)?;
        let out = block.forward(x, temb, freqs_cos, freqs_sin)?;
        if std::env::var("FLUX_TRACE").is_ok() {
            let rms = |t: &Tensor| -> f32 {
                let f = t.to_dtype(candle_core::DType::F32).unwrap().flatten_all().unwrap();
                if let Ok(v) = f.to_vec1::<f32>() {
                    let m = v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>() / v.len() as f64;
                    m.sqrt() as f32
                } else { 0.0 }
            };
            eprintln!("    [TRACE] single.{block_idx} in={:.4} out={:.4}", rms(x), rms(&out));
        }
        Ok(out)
    }

    /// Execute all DoubleStreamBlocks with Asynchronous Double-Buffering Prefetcher (Masquage I/O + déquantisation FP8).
    pub fn execute_double_blocks_pipelined(
        &self,
        num_blocks: usize,
        mut img: Tensor,
        mut txt: Tensor,
        temb: &Tensor,
        img_freqs_cos: Option<&Tensor>,
        img_freqs_sin: Option<&Tensor>,
        txt_freqs_cos: Option<&Tensor>,
        txt_freqs_sin: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        if num_blocks == 0 {
            return Ok((img, txt));
        }

        // Fast path: if all blocks are already cached in VRAM, execute sequentially with zero channel overhead
        if self.max_cached_double_blocks >= num_blocks {
            let all_cached = {
                if let Ok(c) = self.double_blocks_cache.lock() {
                    (0..num_blocks).all(|i| c.contains_key(&i))
                } else { false }
            };
            if all_cached {
                for i in 0..num_blocks {
                    let (next_img, next_txt) = self.execute_double_block(
                        i,
                        &img,
                        &txt,
                        temb,
                        img_freqs_cos,
                        img_freqs_sin,
                        txt_freqs_cos,
                        txt_freqs_sin,
                    )?;
                    img = next_img;
                    txt = next_txt;
                }
                return Ok((img, txt));
            }
        }

        // Spawn background prefetch thread with a bounded ring channel (depth 2 for smooth double-buffering)
        let (tx, rx): (SyncSender<Result<(usize, HashMap<String, Tensor>)>>, Receiver<Result<(usize, HashMap<String, Tensor>)>>) = sync_channel(2);
        let archive = self.archive.clone();
        let device = self.device.clone();
        let dtype = self.dtype;
        let hidden_dim = self.hidden_dim;
        let family = self.family;
        let lora_deltas = self.lora_deltas.clone();
        let double_blocks_cache = self.double_blocks_cache.clone();
        let max_cached = self.max_cached_double_blocks;

        let prefetch_handle: JoinHandle<()> = thread::spawn(move || {
            let dummy_streamer = SequentialBlockStreamer {
                archive,
                device,
                dtype,
                hidden_dim,
                num_heads: 0,
                mlp_ratio: 0,
                family,
                lora_deltas,
                double_blocks_cache,
                single_blocks_cache: Arc::new(Mutex::new(HashMap::new())),
                max_cached_double_blocks: max_cached,
                max_cached_single_blocks: 0,
            };

            for idx in 0..num_blocks {
                let res = dummy_streamer.load_double_block_tensors(idx).map(|t| (idx, t));
                if tx.send(res).is_err() {
                    break; // Main thread dropped receiver
                }
            }
        });

        for i in 0..num_blocks {
            let (idx, tensors) = rx.recv()
                .map_err(|e| candle_core::Error::Msg(format!("Double-buffering channel error: {e}")))?
                .map_err(|e| candle_core::Error::Msg(format!("Prefetch error at double block {i}: {e}")))?;

            assert_eq!(idx, i, "Prefetch pipeline block index mismatch");

            let vb = VarBuilder::from_tensors(tensors, self.dtype, &self.device);
            let block = DoubleStreamBlock::new(self.hidden_dim, self.num_heads, self.mlp_ratio, vb)?;
            let (next_img, next_txt) = block.forward(
                &img,
                &txt,
                temb,
                img_freqs_cos,
                img_freqs_sin,
                txt_freqs_cos,
                txt_freqs_sin,
            )?;
            img = next_img;
            txt = next_txt;

            if std::env::var("FLUX_TRACE").is_ok() {
                let f = img.to_dtype(candle_core::DType::F32).unwrap().flatten_all().unwrap();
                if let Ok(v) = f.to_vec1::<f32>() {
                    let m = v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>() / v.len() as f64;
                    eprintln!("    [TRACE] after pipelined double.{i} img_rms={:.5}", m.sqrt());
                }
            }
        }

        let _ = prefetch_handle.join();
        Ok((img, txt))
    }

    /// Execute all SingleStreamBlocks with Asynchronous Double-Buffering Prefetcher (Masquage I/O + déquantisation FP8).
    pub fn execute_single_blocks_pipelined(
        &self,
        num_blocks: usize,
        mut unified: Tensor,
        temb: &Tensor,
        freqs_cos: Option<&Tensor>,
        freqs_sin: Option<&Tensor>,
    ) -> Result<Tensor> {
        if num_blocks == 0 {
            return Ok(unified);
        }

        // Fast path: if all blocks are already cached in VRAM, execute sequentially with zero channel overhead
        if self.max_cached_single_blocks >= num_blocks {
            let all_cached = {
                if let Ok(c) = self.single_blocks_cache.lock() {
                    (0..num_blocks).all(|i| c.contains_key(&i))
                } else { false }
            };
            if all_cached {
                for i in 0..num_blocks {
                    unified = self.execute_single_block(i, &unified, temb, freqs_cos, freqs_sin)?;
                }
                return Ok(unified);
            }
        }

        // Spawn background prefetch thread with a bounded ring channel (depth 2 for smooth double-buffering)
        let (tx, rx): (SyncSender<Result<(usize, HashMap<String, Tensor>)>>, Receiver<Result<(usize, HashMap<String, Tensor>)>>) = sync_channel(2);
        let archive = self.archive.clone();
        let device = self.device.clone();
        let dtype = self.dtype;
        let hidden_dim = self.hidden_dim;
        let family = self.family;
        let lora_deltas = self.lora_deltas.clone();
        let single_blocks_cache = self.single_blocks_cache.clone();
        let max_cached = self.max_cached_single_blocks;

        let prefetch_handle: JoinHandle<()> = thread::spawn(move || {
            let dummy_streamer = SequentialBlockStreamer {
                archive,
                device,
                dtype,
                hidden_dim,
                num_heads: 0,
                mlp_ratio: 0,
                family,
                lora_deltas,
                double_blocks_cache: Arc::new(Mutex::new(HashMap::new())),
                single_blocks_cache,
                max_cached_double_blocks: 0,
                max_cached_single_blocks: max_cached,
            };

            for idx in 0..num_blocks {
                let res = dummy_streamer.load_single_block_tensors(idx).map(|t| (idx, t));
                if tx.send(res).is_err() {
                    break; // Main thread dropped receiver
                }
            }
        });

        for i in 0..num_blocks {
            let (idx, tensors) = rx.recv()
                .map_err(|e| candle_core::Error::Msg(format!("Double-buffering channel error: {e}")))?
                .map_err(|e| candle_core::Error::Msg(format!("Prefetch error at single block {i}: {e}")))?;

            assert_eq!(idx, i, "Prefetch pipeline block index mismatch");

            let vb = VarBuilder::from_tensors(tensors, self.dtype, &self.device);
            let block = SingleStreamBlock::new(self.hidden_dim, self.num_heads, self.mlp_ratio, vb)?;
            unified = block.forward(&unified, temb, freqs_cos, freqs_sin)?;

            if std::env::var("FLUX_TRACE").is_ok() {
                let rms = |t: &Tensor| -> f32 {
                    let f = t.to_dtype(candle_core::DType::F32).unwrap().flatten_all().unwrap();
                    if let Ok(v) = f.to_vec1::<f32>() {
                        let m = v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>() / v.len() as f64;
                        m.sqrt() as f32
                    } else { 0.0 }
                };
                eprintln!("    [TRACE] pipelined single.{i} out_rms={:.4}", rms(&unified));
            }
        }

        let _ = prefetch_handle.join();
        Ok(unified)
    }
}

/// Combine a Diffusers-layout split QKV (`{base}@Q.weight`, `@K`, `@V`) into a single fused
/// `{base}.weight` (concatenated along dim 0). Removes the split entries so the block builder sees a
/// single fused linear weight, as it expects.
fn fuse_split_qkv(tensors: &mut HashMap<String, Tensor>, base: &str) {
    let get = |tag: &str| tensors.get(&format!("{base}@{tag}.weight")).cloned();
    let (q, k, v) = (get("Q"), get("K"), get("V"));
    let (q, k, v) = match (q, k, v) {
        (Some(q), Some(k), Some(v)) => (q, k, v),
        _ => return, // not split / partial; leave as-is
    };
    if let Ok(fused) = Tensor::cat(&[&q, &k, &v], 0) {
        tensors.insert(format!("{base}.weight"), fused);
        tensors.remove(&format!("{base}@Q.weight"));
        tensors.remove(&format!("{base}@K.weight"));
        tensors.remove(&format!("{base}@V.weight"));
    }
}
