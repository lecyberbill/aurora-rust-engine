// [WFGY] Zone: SAFE | λ: 0.35 | Fallbacks: 1 (OpenCLIP in_proj tensor splitting) | Action: OpenCLIP in_proj splitting, key mapping and text_projection support

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use half::{bf16, f16};
use memmap2::Mmap;
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use crate::error::{LuminaError, Result};

/// Unified, format-agnostic access to model weights.
///
/// Every brick (text encoders, DiT transformer, VAE) consumes this trait rather than a concrete
/// format struct. Implementations exist for single & multi-file **safetensors** archives and can be
/// added for **GGUF** (llama.cpp) or any other container — so the caller assembles interchangeable
/// parts without touching per-format code.
pub trait WeightsSource: Send + Sync {
    /// Decode a named tensor onto `device` at `dtype`. FP8 / NVFP4 / scaled-FP8 dequantisation is the
    /// responsibility of the implementation.
    fn get_tensor(&self, name: &str, device: &Device, dtype: DType) -> Result<Tensor>;

    /// Whether a tensor with this exact name exists.
    fn contains(&self, name: &str) -> bool;

    /// Raw (stored) dtype and shape, without decoding. Useful for architecture detection.
    fn raw_info(&self, name: &str) -> Option<(safetensors::Dtype, Vec<usize>)>;

    /// All tensor names in this source.
    fn keys(&self) -> Vec<String>;

    /// Short human-readable description (e.g. "safetensors xN shards", "gguf").
    fn describe(&self) -> String {
        "weights".to_string()
    }
}

/// Blanket impl so any `WeightsSource` can be consumed by reference (e.g. `&dyn WeightsSource`).
impl<'a> WeightsSource for &'a dyn WeightsSource {
    fn get_tensor(&self, name: &str, device: &Device, dtype: DType) -> Result<Tensor> {
        (**self).get_tensor(name, device, dtype)
    }
    fn contains(&self, name: &str) -> bool { (**self).contains(name) }
    fn raw_info(&self, name: &str) -> Option<(safetensors::Dtype, Vec<usize>)> { (**self).raw_info(name) }
    fn keys(&self) -> Vec<String> { (**self).keys() }
    fn describe(&self) -> String { (**self).describe() }
}

pub struct SafeTensorsArchive {
    // One mmap per shard file; a single-file archive simply contains one entry.
    _mmaps: Vec<Arc<Mmap>>,
    // Tensor name -> (dtype, shape, shard_idx, offset_into_shard, byte_len)
    tensors: HashMap<String, (safetensors::Dtype, Vec<usize>, usize, usize, usize)>,
}

unsafe impl Send for SafeTensorsArchive {}
unsafe impl Sync for SafeTensorsArchive {}

impl SafeTensorsArchive {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::open_shards(&[path.as_ref().to_path_buf()])
    }

    /// Open a set of shard files (HF multi-file checkpoints) as a single logical archive.
    /// Each tensor must live entirely in exactly one shard; the archive resolves which.
    pub fn open_shards<P: AsRef<Path>>(paths: &[P]) -> Result<Self> {
        if paths.is_empty() {
            return Err(LuminaError::Config("open_shards: no shard paths provided".into()));
        }

        let mut mmaps = Vec::with_capacity(paths.len());
        let mut tensors = HashMap::new();

        for (shard_idx, p) in paths.iter().enumerate() {
            let file = File::open(p)?;
            let mmap = unsafe { Mmap::map(&file)? };
            let mmap_arc = Arc::new(mmap);
            let raw_data = mmap_arc.as_ptr();
            let bytes: &[u8] = &mmap_arc;

            let st = SafeTensors::deserialize(bytes)?;
            for (name, view) in st.tensors() {
                let data_offset = view.data().as_ptr() as usize - raw_data as usize;
                let data_len = view.data().len();
                let meta = (view.dtype(), view.shape().to_vec(), shard_idx, data_offset, data_len);
                let dup = tensors.insert(name.clone(), meta).is_some();
                if dup {
                    return Err(LuminaError::Config(format!(
                        "open_shards: duplicate tensor '{}' across shards", name
                    )));
                }
            }
            mmaps.push(mmap_arc);
        }

        Ok(Self { _mmaps: mmaps, tensors })
    }

    /// Open every `*.safetensors` shard found in a directory (HuggingFace multi-file layout),
    /// sorted by name for deterministic shard ordering. Returns an error if none are found.
    pub fn open_shards_dir<P: AsRef<Path>>(dir: P) -> Result<Self> {
        let mut shards: Vec<PathBuf> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("safetensors"))
            .collect();
        shards.sort();
        if shards.is_empty() {
            return Err(LuminaError::Config("open_shards_dir: no .safetensors shards found".into()));
        }
        Self::open_shards(&shards)
    }

    pub fn tensor_names(&self) -> Vec<String> {
        self.tensors.keys().cloned().collect()
    }

    pub fn get_tensor(&self, name: &str, device: &Device, dtype: DType) -> Result<Tensor> {
        let (st_dtype, shape, shard_idx, offset, len) = self
            .tensors
            .get(name)
            .ok_or_else(|| LuminaError::MissingWeight(format!("Tensor '{}' not found in archive", name)))?;

        let shard = self._mmaps.get(*shard_idx)
            .ok_or_else(|| LuminaError::Config(format!("Shard index {} not present", shard_idx)))?;
        let raw_data = shard.as_ptr();
        let slice = unsafe { std::slice::from_raw_parts(raw_data.add(*offset), *len) };
        let tensor = match st_dtype {
            safetensors::Dtype::F32 => {
                let data: &[f32] = bytemuck_cast_slice(slice);
                Tensor::from_slice(data, shape.as_slice(), device)?
            }
            safetensors::Dtype::F16 => {
                let data: &[f16] = bytemuck_cast_slice(slice);
                Tensor::from_slice(data, shape.as_slice(), device)?
            }
            safetensors::Dtype::BF16 => {
                let data: &[bf16] = bytemuck_cast_slice(slice);
                Tensor::from_slice(data, shape.as_slice(), device)?
            }
            safetensors::Dtype::I64 => {
                let data: &[i64] = bytemuck_cast_slice(slice);
                Tensor::from_slice(data, shape.as_slice(), device)?
            }
            safetensors::Dtype::I32 => {
                let data: &[i32] = bytemuck_cast_slice(slice);
                let data_i64: Vec<i64> = data.iter().map(|&x| x as i64).collect();
                Tensor::from_vec(data_i64, shape.as_slice(), device)?
            }
            safetensors::Dtype::U32 => {
                let data: &[u32] = bytemuck_cast_slice(slice);
                Tensor::from_slice(data, shape.as_slice(), device)?
            }
            safetensors::Dtype::U8 => {
                let data: &[u8] = slice;
                Tensor::from_slice(data, shape.as_slice(), device)?
            }
            safetensors::Dtype::F8_E4M3 => {
                let data: &[u8] = slice;
                let lut = get_fp8_e4m3_lut();
                let f16_data: Vec<half::f16> = data.iter().map(|&b| lut[b as usize]).collect();
                let raw_tensor = Tensor::from_vec(f16_data, shape.as_slice(), device)?;
                if dtype == DType::F16 {
                    raw_tensor
                } else {
                    raw_tensor.to_dtype(dtype)?
                }
            }
            safetensors::Dtype::F8_E5M2 => {
                let data: &[u8] = slice;
                let lut = get_fp8_e5m2_lut();
                let f16_data: Vec<half::f16> = data.iter().map(|&b| lut[b as usize]).collect();
                let raw_tensor = Tensor::from_vec(f16_data, shape.as_slice(), device)?;
                if dtype == DType::F16 {
                    raw_tensor
                } else {
                    raw_tensor.to_dtype(dtype)?
                }
            }
            other => {
                return Err(LuminaError::Config(format!(
                    "Unsupported SafeTensors dtype {:?} for tensor {}",
                    other, name
                )))
            }
        };

        // Support Scaled FP8 checkpoints (e.g. Klein-9B and Flux2-Dev scaled FP8):
        // If a weight has a corresponding `{name}_scale` or `{name}.weight_scale` or `{name}.scale_weight`
        let scaled_tensor = if let Some(base_name) = name.strip_suffix(".weight") {
            let scale_key_1 = format!("{}.weight_scale", base_name);
            let scale_key_2 = format!("{}.scale_weight", base_name);
            let scale_key_3 = format!("{}_scale", name);
            let found_scale = if self.tensors.contains_key(&scale_key_1) {
                Some(scale_key_1)
            } else if self.tensors.contains_key(&scale_key_2) {
                Some(scale_key_2)
            } else if self.tensors.contains_key(&scale_key_3) {
                Some(scale_key_3)
            } else {
                None
            };

            if let Some(scale_key) = found_scale {
                let (s_dtype, s_shape, s_shard_idx, s_offset, s_len) = &self.tensors[&scale_key];
                let s_shard = self._mmaps.get(*s_shard_idx).ok_or_else(|| LuminaError::Config("shard missing".into()))?;
                let s_slice = unsafe { std::slice::from_raw_parts(s_shard.as_ptr().add(*s_offset), *s_len) };
                let scale_val = match s_dtype {
                    safetensors::Dtype::F32 => {
                        let data: &[f32] = bytemuck_cast_slice(s_slice);
                        Tensor::from_slice(data, s_shape.as_slice(), device)?
                    }
                    safetensors::Dtype::F16 => {
                        let data: &[f16] = bytemuck_cast_slice(s_slice);
                        Tensor::from_slice(data, s_shape.as_slice(), device)?.to_dtype(DType::F32)?
                    }
                    safetensors::Dtype::BF16 => {
                        let data: &[bf16] = bytemuck_cast_slice(s_slice);
                        Tensor::from_slice(data, s_shape.as_slice(), device)?.to_dtype(DType::F32)?
                    }
                    _ => Tensor::ones((1,), DType::F32, device)?,
                };
                let tensor_f32 = tensor.to_dtype(DType::F32)?;
                let scaled = tensor_f32.broadcast_mul(&scale_val)?;
                scaled.to_dtype(dtype)?
            } else {
                tensor
            }
        } else {
            tensor
        };

        if (scaled_tensor.dtype() == DType::F32 || scaled_tensor.dtype() == DType::F16 || scaled_tensor.dtype() == DType::BF16)
            && scaled_tensor.dtype() != dtype
        {
            Ok(scaled_tensor.to_dtype(dtype)?)
        } else {
            Ok(scaled_tensor)
        }
    }

    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// Return the raw (stored) dtype and shape of a tensor without decoding it.
    /// Useful to distinguish FP8 / NVFP4 / BF16 checkpoints for diagnostics.
    pub fn raw_info(&self, name: &str) -> Option<(safetensors::Dtype, Vec<usize>)> {
        self.tensors.get(name).map(|(d, s, _, _, _)| (*d, s.clone()))
    }

    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.tensors.keys()
    }
}

fn bytemuck_cast_slice<T>(bytes: &[u8]) -> &[T] {
    let elem_size = std::mem::size_of::<T>();
    assert_eq!(
        bytes.len() % elem_size,
        0,
        "Byte buffer size must be multiple of element size"
    );
    unsafe {
        std::slice::from_raw_parts(bytes.as_ptr() as *const T, bytes.len() / elem_size)
    }
}

static FP8_E4M3_LUT_F16: std::sync::OnceLock<[half::f16; 256]> = std::sync::OnceLock::new();
static FP8_E5M2_LUT_F16: std::sync::OnceLock<[half::f16; 256]> = std::sync::OnceLock::new();

fn get_fp8_e4m3_lut() -> &'static [half::f16; 256] {
    FP8_E4M3_LUT_F16.get_or_init(|| {
        let mut lut = [half::f16::ZERO; 256];
        for b in 0..=255u8 {
            lut[b as usize] = half::f16::from_f32(fp8_e4m3_to_f32(b));
        }
        lut
    })
}

fn get_fp8_e5m2_lut() -> &'static [half::f16; 256] {
    FP8_E5M2_LUT_F16.get_or_init(|| {
        let mut lut = [half::f16::ZERO; 256];
        for b in 0..=255u8 {
            lut[b as usize] = half::f16::from_f32(fp8_e5m2_to_f32(b));
        }
        lut
    })
}

/// Convert FP8 E4M3FN (OCP Standard) byte to F32
fn fp8_e4m3_to_f32(byte: u8) -> f32 {
    let sign = (byte >> 7) & 1;
    let exp = (byte >> 3) & 0x0F;
    let mant = byte & 0x07;

    if exp == 0 && mant == 0 {
        return if sign == 1 { -0.0 } else { 0.0 };
    }
    if exp == 0x0F && mant == 0x07 {
        return f32::NAN; // E4M3FN NaN
    }

    let val = if exp == 0 {
        // Subnormal: 2^(-6) * (mant / 8)
        (mant as f32 / 8.0) * (2.0f32).powi(-6)
    } else {
        // Normal: 2^(exp - 7) * (1 + mant / 8)
        (1.0 + mant as f32 / 8.0) * (2.0f32).powi(exp as i32 - 7)
    };

    if sign == 1 { -val } else { val }
}

/// Convert FP8 E5M2 byte to F32
fn fp8_e5m2_to_f32(byte: u8) -> f32 {
    let sign = (byte >> 7) & 1;
    let exp = (byte >> 2) & 0x1F;
    let mant = byte & 0x03;

    if exp == 0 && mant == 0 {
        return if sign == 1 { -0.0 } else { 0.0 };
    }
    if exp == 0x1F {
        return if mant == 0 {
            if sign == 1 { f32::NEG_INFINITY } else { f32::INFINITY }
        } else {
            f32::NAN
        };
    }

    let val = if exp == 0 {
        // Subnormal: 2^(-14) * (mant / 4)
        (mant as f32 / 4.0) * (2.0f32).powi(-14)
    } else {
        // Normal: 2^(exp - 15) * (1 + mant / 4)
        (1.0 + mant as f32 / 4.0) * (2.0f32).powi(exp as i32 - 15)
    };

    if sign == 1 { -val } else { val }
}

pub struct WeightRouter<'a> {
    archive: &'a SafeTensorsArchive,
    device: Device,
    dtype: DType,
    lora_deltas: HashMap<String, Tensor>,
}

impl<'a> WeightRouter<'a> {
    pub fn new(archive: &'a SafeTensorsArchive, device: Device, dtype: DType) -> Self {
        Self {
            archive,
            device,
            dtype,
            lora_deltas: HashMap::new(),
        }
    }

    pub fn set_lora_deltas(&mut self, deltas: HashMap<String, Tensor>) {
        self.lora_deltas = deltas;
    }

    pub fn add_lora_deltas(&mut self, deltas: &[(String, Tensor)]) -> Result<()> {
        for (name, delta) in deltas {
            if let Some(existing) = self.lora_deltas.get_mut(name) {
                *existing = (existing.as_ref() + delta)?;
            } else {
                self.lora_deltas.insert(name.clone(), delta.clone());
            }
        }
        Ok(())
    }

    pub fn clear_lora_deltas(&mut self) {
        self.lora_deltas.clear();
    }

    fn apply_delta_if_present(&self, prefix: &str, key: &str, tensor: Tensor, target_device: &Device, target_dtype: DType) -> Result<Tensor> {
        let lookup_key = if !key.ends_with(".weight") && !key.ends_with(".bias") {
            format!("{}.weight", key)
        } else {
            key.to_string()
        };
        let namespaced = if !prefix.is_empty() {
            format!("{}.{}", prefix, lookup_key)
        } else {
            lookup_key.clone()
        };

        if let Some(delta) = self.lora_deltas.get(&namespaced)
            .or_else(|| self.lora_deltas.get(&lookup_key))
            .or_else(|| self.lora_deltas.get(key))
        {
            let delta = delta.to_device(target_device)?.to_dtype(target_dtype)?;
            if delta.shape() == tensor.shape() {
                return Ok((&tensor + &delta)?);
            }
        }
        Ok(tensor)
    }

    /// Apply LoRA deltas to a single transformer weight, handling the fused QKV projection.
    ///
    /// Flux MMDiT stores Q,K,V in ONE linear (`img_attn.qkv.weight`, shape `[3*dim, in]`). LoRAs may
    /// target a single projection (`attn.to_q`, `.to_k`, `.to_v`), producing a delta of shape
    /// `[dim, in]`. Such deltas carry a `@Q` / `@K` / `@V` tag in their key. A plain delta (shape
    /// `[3*dim, in]`, BFL format) is applied directly. This reconciles both.
    pub fn apply_flux_delta_to_tensor(
        &self,
        key: &str,
        tensor: Tensor,
        target_device: &Device,
        target_dtype: DType,
    ) -> Result<Tensor> {
        apply_flux_deltas_to_tensor(&self.lora_deltas, key, tensor, target_device, target_dtype)
    }
}

/// Standalone helper used by the sequential streamer (which has `&HashMap<String, Tensor>` rather
/// than a `&WeightRouter`) to splice LoRA deltas into a transformer weight during per-block loading.
pub fn apply_flux_deltas_to_tensor(
    deltas: &HashMap<String, Tensor>,
    key: &str,
    tensor: Tensor,
    target_device: &Device,
    target_dtype: DType,
) -> Result<Tensor> {
    let weight_key = if !key.ends_with(".weight") { format!("{key}.weight") } else { key.to_string() };
    let weight_key_stripped = weight_key.strip_suffix(".weight").unwrap_or(&weight_key).to_string();

    // Plain (already fused / BFL) delta that matches the whole tensor.
    if let Some(delta) = deltas.get(&weight_key) {
        let d = delta.to_device(target_device)?.to_dtype(target_dtype)?;
        if d.shape() == tensor.shape() {
            return Ok((&tensor + &d)?);
        }
    }

    // Sliced QKV deltas. Rows-sliced (`@Q`/`@K`/`@V`, shape [slab, in]) target a fused projection
    // (Q|K|V stacked on rows, e.g. `*.qkv` = [3*dim, in] or single-stream `linear1` = [9*dim, in]).
    let base0 = tensor.to_device(target_device)?.to_dtype(target_dtype)?;
    if base0.dims().len() != 2 {
        return Ok(tensor); // only 2D Linear/Conv weights are spliced
    }
    let (out, cols) = (base0.dim(0)?, base0.dim(1)?);

    let row_slab = ["@Q", "@K", "@V"].iter().find_map(|&t| deltas.get(&format!("{weight_key_stripped}{t}.weight")).map(|d| d.dim(0).unwrap_or(0)));

    if let Some(slab) = row_slab {
        if out < slab * 3 { return Ok(tensor); }
        // Preserve remaining rows (e.g. single-stream linear1 = [Q|K|V|MLP] has 9*slab rows); rebuild
        // with Q,K,V spliced and any extra rows untouched.
        let build_slab = |tag: &str, base: &Tensor| -> Result<Tensor> {
            let tagged = format!("{weight_key_stripped}{tag}.weight");
            let existing = match tag {
                "@Q" => base.narrow(0, 0, slab).map_err(LuminaError::Candle)?,
                "@K" => base.narrow(0, slab, slab).map_err(LuminaError::Candle)?,
                _ => base.narrow(0, slab * 2, slab).map_err(LuminaError::Candle)?,
            };
            if let Some(delta) = deltas.get(&tagged) {
                let d = delta.to_device(target_device).map_err(LuminaError::Candle)?.to_dtype(target_dtype).map_err(LuminaError::Candle)?;
                if d.dims() == [slab, cols] {
                    Ok((&existing + &d).map_err(LuminaError::Candle)?)
                } else {
                    Ok(existing)
                }
            } else {
                Ok(existing)
            }
        };
        let q = build_slab("@Q", &base0)?;
        let k = build_slab("@K", &base0)?;
        let v = build_slab("@V", &base0)?;
        if out == slab * 3 {
            return Tensor::cat(&[&q, &k, &v], 0).map_err(LuminaError::Candle);
        }
        let tail = base0.narrow(0, slab * 3, out - slab * 3).map_err(LuminaError::Candle)?;
        Tensor::cat(&[&q, &k, &v, &tail], 0).map_err(LuminaError::Candle);
    }

    Ok(tensor)
}

impl<'a> WeightRouter<'a> {
    pub fn var_builder_for_prefix(&self, primary_prefix: &str, alt_prefixes: &[&str]) -> Result<VarBuilder<'static>> {
        let mut tensors = HashMap::new();

        for key in self.archive.keys() {
            let matched_suffix = if key.starts_with(primary_prefix) {
                Some(&key[primary_prefix.len()..])
            } else {
                alt_prefixes
                    .iter()
                    .find(|prefix| key.starts_with(**prefix))
                    .map(|prefix| &key[prefix.len()..])
            };

            if let Some(suffix) = matched_suffix {
                let tensor = self.archive.get_tensor(key, &self.device, self.dtype)?;
                let final_tensor = self.apply_delta_if_present("", suffix, tensor, &self.device, self.dtype)?;
                tensors.insert(suffix.to_string(), final_tensor);
            }
        }

        if tensors.is_empty() {
            return Err(LuminaError::MissingWeight(format!(
                "No weights found matching prefix '{}' or alternatives {:?}",
                primary_prefix, alt_prefixes
            )));
        }

        Ok(VarBuilder::from_tensors(tensors, self.dtype, &self.device))
    }

    pub fn flux_header_var_builder(&self) -> Result<VarBuilder<'static>> {
        let mut tensors = HashMap::new();
        let header_prefixes = ["img_in.", "txt_in.", "time_in.", "vector_in.", "guidance_in.", "final_layer."];

        for key in self.archive.keys() {
            for prefix in &header_prefixes {
                if key.starts_with(prefix) {
                    let tensor = self.archive.get_tensor(key, &self.device, self.dtype)?;
                    tensors.insert(key.clone(), tensor);
                    break;
                } else if let Some(stripped) = key.strip_prefix("model.diffusion_model.") {
                    if stripped.starts_with(prefix) {
                        let tensor = self.archive.get_tensor(key, &self.device, self.dtype)?;
                        tensors.insert(stripped.to_string(), tensor);
                        break;
                    }
                }
            }
        }

        if tensors.is_empty() {
            return Err(LuminaError::MissingWeight("No Flux.1 header weights found in checkpoint".to_string()));
        }

        Ok(VarBuilder::from_tensors(tensors, self.dtype, &self.device))
    }

    pub fn flux_var_builder(&self) -> Result<VarBuilder<'static>> {
        let mut tensors = HashMap::new();
        let flux_prefixes = ["double_blocks.", "single_blocks.", "img_in.", "txt_in.", "time_in.", "vector_in.", "guidance_in.", "final_layer."];

        for key in self.archive.keys() {
            for prefix in &flux_prefixes {
                if key.starts_with(prefix) {
                    let tensor = self.archive.get_tensor(key, &self.device, self.dtype)?;
                    tensors.insert(key.clone(), tensor);
                    break;
                } else if let Some(stripped) = key.strip_prefix("model.diffusion_model.") {
                    if stripped.starts_with(prefix) {
                        let tensor = self.archive.get_tensor(key, &self.device, self.dtype)?;
                        tensors.insert(stripped.to_string(), tensor);
                        break;
                    }
                }
            }
        }

        if tensors.is_empty() {
            return Err(LuminaError::MissingWeight("No Flux.1 / MMDiT weights found in checkpoint".to_string()));
        }

        Ok(VarBuilder::from_tensors(tensors, self.dtype, &self.device))
    }

    pub fn unet_var_builder(&self) -> Result<VarBuilder<'static>> {
        let mut tensors = HashMap::new();
        let unet_prefix = "model.diffusion_model.";

        for key in self.archive.keys() {
            if let Some(compvis_key) = key.strip_prefix(unet_prefix) {
                let diffusers_key = translate_compvis_unet_key(compvis_key);
                let tensor = self.archive.get_tensor(key, &self.device, self.dtype)?;
                let final_tensor = self.apply_delta_if_present("unet", &diffusers_key, tensor, &self.device, self.dtype)?;
                tensors.insert(diffusers_key, final_tensor);
            } else if key.starts_with("conv_in.") || key.starts_with("down_blocks.") || key.starts_with("mid_block.") || key.starts_with("up_blocks.") {
                let tensor = self.archive.get_tensor(key, &self.device, self.dtype)?;
                let final_tensor = self.apply_delta_if_present("unet", key, tensor, &self.device, self.dtype)?;
                tensors.insert(key.clone(), final_tensor);
            }
        }

        if tensors.is_empty() {
            return Err(LuminaError::MissingWeight("No UNet weights found in checkpoint".to_string()));
        }

        Ok(VarBuilder::from_tensors(tensors, self.dtype, &self.device))
    }

    pub fn vae_var_builder(&self) -> Result<VarBuilder<'static>> {
        let mut tensors = HashMap::new();
        let vae_prefix = "first_stage_model.";

        for key in self.archive.keys() {
            if let Some(compvis_key) = key.strip_prefix(vae_prefix) {
                let diffusers_key = translate_compvis_vae_key(compvis_key);
                let tensor = self.archive.get_tensor(key, &self.device, self.dtype)?;
                tensors.insert(diffusers_key, tensor);
            } else if let Some(stripped) = key.strip_prefix("vae.") {
                let tensor = self.archive.get_tensor(key, &self.device, self.dtype)?;
                tensors.insert(stripped.to_string(), tensor);
            } else if key.starts_with("encoder.") || key.starts_with("decoder.") || key.starts_with("quant_conv.") || key.starts_with("post_quant_conv.") || key.starts_with("bn.") {
                let tensor = self.archive.get_tensor(key, &self.device, self.dtype)?;
                tensors.insert(key.clone(), tensor);
            }
        }

        if tensors.is_empty() {
            return Err(LuminaError::MissingWeight("No VAE weights found in checkpoint".to_string()));
        }

        Ok(VarBuilder::from_tensors(tensors, self.dtype, &self.device))
    }

    pub fn vae_var_builder_f32(&self) -> Result<VarBuilder<'static>> {
        let mut tensors = HashMap::new();
        let vae_prefix = "first_stage_model.";

        for key in self.archive.keys() {
            if let Some(compvis_key) = key.strip_prefix(vae_prefix) {
                let diffusers_key = translate_compvis_vae_key(compvis_key);
                let tensor = self.archive.get_tensor(key, &self.device, DType::F32)?;
                tensors.insert(diffusers_key, tensor);
            } else if key.starts_with("encoder.") || key.starts_with("decoder.") || key.starts_with("post_quant_conv.") {
                let tensor = self.archive.get_tensor(key, &self.device, DType::F32)?;
                tensors.insert(key.clone(), tensor);
            }
        }

        if tensors.is_empty() {
            return Err(LuminaError::MissingWeight("No VAE weights found in checkpoint".to_string()));
        }

        Ok(VarBuilder::from_tensors(tensors, DType::F32, &self.device))
    }

    pub fn clip_l_var_builder(&self) -> Result<VarBuilder<'static>> {
        self.clip_l_var_builder_on_device(&self.device, self.dtype)
    }

    pub fn clip_l_var_builder_on_device(&self, target_device: &Device, target_dtype: DType) -> Result<VarBuilder<'static>> {
        let mut tensors = HashMap::new();
        let prefixes = [
            "text_encoders.clip_l.transformer.",
            "text_encoders.clip_l.",
            "conditioner.embedders.0.transformer.",
            "conditioner.embedders.0.",
            "cond_stage_model.transformer.",
            "cond_stage_model.",
            "text_encoder.",
        ];

        for key in self.archive.keys() {
            for prefix in &prefixes {
                if let Some(suffix) = key.strip_prefix(*prefix) {
                    let mut base = suffix.to_string();
                    if base.starts_with("transformer.") {
                        base = base.strip_prefix("transformer.").unwrap().to_string();
                    }

                    let raw_core = if base.starts_with("text_model.") {
                        base.strip_prefix("text_model.").unwrap().to_string()
                    } else {
                        base.clone()
                    };

                    let tensor = self.archive.get_tensor(key, target_device, target_dtype)?;
                    let final_tensor = self.apply_delta_if_present("te1", &format!("text_model.{}", raw_core), tensor, target_device, target_dtype)?;
                    tensors.insert(raw_core.clone(), final_tensor.clone());
                    tensors.insert(format!("text_model.{}", raw_core), final_tensor.clone());
                    tensors.insert(format!("transformer.text_model.{}", raw_core), final_tensor);
                    break;
                }
            }
        }

        if tensors.is_empty() {
            return Err(LuminaError::MissingWeight("No CLIP-L weights found in checkpoint".to_string()));
        }

        Ok(VarBuilder::from_tensors(tensors, target_dtype, target_device))
    }

    pub fn t5xxl_var_builder_on_device(&self, target_device: &Device, target_dtype: DType) -> Result<VarBuilder<'static>> {
        let mut tensors = HashMap::new();
        let prefixes = [
            "text_encoders.t5xxl.transformer.",
            "text_encoders.t5xxl.",
            "conditioner.embedders.1.transformer.",
            "conditioner.embedders.1.",
            "t5xxl.",
        ];

        for key in self.archive.keys() {
            for prefix in &prefixes {
                if let Some(suffix) = key.strip_prefix(*prefix) {
                    let tensor = self.archive.get_tensor(key, target_device, target_dtype)?;
                    tensors.insert(suffix.to_string(), tensor);
                    break;
                }
            }
        }

        if tensors.is_empty() {
            return Err(LuminaError::MissingWeight("No T5-XXL weights found in checkpoint".to_string()));
        }

        Ok(VarBuilder::from_tensors(tensors, target_dtype, target_device))
    }

    pub fn open_clip_g_var_builder(&self) -> Result<VarBuilder<'static>> {
        self.open_clip_g_var_builder_on_device(&self.device, self.dtype)
    }

    pub fn open_clip_g_var_builder_on_device(&self, target_device: &Device, target_dtype: DType) -> Result<VarBuilder<'static>> {
        let mut tensors = HashMap::new();
        let prefix = "conditioner.embedders.1.model.";

        for key in self.archive.keys() {
            if let Some(suffix) = key.strip_prefix(prefix) {
                let tensor = self.archive.get_tensor(key, target_device, target_dtype)?;

                if suffix == "token_embedding.weight" {
                    tensors.insert("text_model.embeddings.token_embedding.weight".to_string(), tensor.clone());
                    tensors.insert("embeddings.token_embedding.weight".to_string(), tensor.clone());
                } else if suffix == "positional_embedding" {
                    tensors.insert("text_model.embeddings.position_embedding.weight".to_string(), tensor.clone());
                    tensors.insert("embeddings.position_embedding.weight".to_string(), tensor.clone());
                } else if suffix == "ln_final.weight" {
                    tensors.insert("text_model.final_layer_norm.weight".to_string(), tensor.clone());
                    tensors.insert("final_layer_norm.weight".to_string(), tensor.clone());
                } else if suffix == "ln_final.bias" {
                    tensors.insert("text_model.final_layer_norm.bias".to_string(), tensor.clone());
                    tensors.insert("final_layer_norm.bias".to_string(), tensor.clone());
                } else if suffix == "text_projection" {
                    tensors.insert("text_projection".to_string(), tensor.clone());
                } else if suffix.starts_with("transformer.resblocks.") {
                    let rest = suffix.strip_prefix("transformer.resblocks.").unwrap();
                    let parts: Vec<&str> = rest.splitn(2, '.').collect();
                    if parts.len() == 2 {
                        let block_idx: usize = parts[0].parse().unwrap_or(0);
                        let subkey = parts[1];

                        if subkey == "attn.in_proj_weight" {
                            let dim = tensor.dim(0)? / 3;
                            let q = tensor.narrow(0, 0, dim)?;
                            let k = tensor.narrow(0, dim, dim)?;
                            let v = tensor.narrow(0, 2 * dim, dim)?;

                            let q = self.apply_delta_if_present("te2", &format!("text_model.encoder.layers.{}.self_attn.q_proj.weight", block_idx), q, target_device, target_dtype)?;
                            let k = self.apply_delta_if_present("te2", &format!("text_model.encoder.layers.{}.self_attn.k_proj.weight", block_idx), k, target_device, target_dtype)?;
                            let v = self.apply_delta_if_present("te2", &format!("text_model.encoder.layers.{}.self_attn.v_proj.weight", block_idx), v, target_device, target_dtype)?;

                            tensors.insert(format!("text_model.encoder.layers.{}.self_attn.q_proj.weight", block_idx), q.clone());
                            tensors.insert(format!("encoder.layers.{}.self_attn.q_proj.weight", block_idx), q);
                            tensors.insert(format!("text_model.encoder.layers.{}.self_attn.k_proj.weight", block_idx), k.clone());
                            tensors.insert(format!("encoder.layers.{}.self_attn.k_proj.weight", block_idx), k);
                            tensors.insert(format!("text_model.encoder.layers.{}.self_attn.v_proj.weight", block_idx), v.clone());
                            tensors.insert(format!("encoder.layers.{}.self_attn.v_proj.weight", block_idx), v);
                        } else if subkey == "attn.in_proj_bias" {
                            let dim = tensor.dim(0)? / 3;
                            let q = tensor.narrow(0, 0, dim)?;
                            let k = tensor.narrow(0, dim, dim)?;
                            let v = tensor.narrow(0, 2 * dim, dim)?;

                            tensors.insert(format!("text_model.encoder.layers.{}.self_attn.q_proj.bias", block_idx), q.clone());
                            tensors.insert(format!("encoder.layers.{}.self_attn.q_proj.bias", block_idx), q);
                            tensors.insert(format!("text_model.encoder.layers.{}.self_attn.k_proj.bias", block_idx), k.clone());
                            tensors.insert(format!("encoder.layers.{}.self_attn.k_proj.bias", block_idx), k);
                            tensors.insert(format!("text_model.encoder.layers.{}.self_attn.v_proj.bias", block_idx), v.clone());
                            tensors.insert(format!("encoder.layers.{}.self_attn.v_proj.bias", block_idx), v);
                        } else if subkey == "attn.out_proj.weight" {
                            let final_tensor = self.apply_delta_if_present("te2", &format!("text_model.encoder.layers.{}.self_attn.out_proj.weight", block_idx), tensor, target_device, target_dtype)?;
                            tensors.insert(format!("text_model.encoder.layers.{}.self_attn.out_proj.weight", block_idx), final_tensor.clone());
                            tensors.insert(format!("encoder.layers.{}.self_attn.out_proj.weight", block_idx), final_tensor);
                        } else if subkey == "attn.out_proj.bias" {
                            tensors.insert(format!("text_model.encoder.layers.{}.self_attn.out_proj.bias", block_idx), tensor.clone());
                            tensors.insert(format!("encoder.layers.{}.self_attn.out_proj.bias", block_idx), tensor.clone());
                        } else if subkey == "ln_1.weight" {
                            tensors.insert(format!("text_model.encoder.layers.{}.layer_norm1.weight", block_idx), tensor.clone());
                            tensors.insert(format!("encoder.layers.{}.layer_norm1.weight", block_idx), tensor.clone());
                        } else if subkey == "ln_1.bias" {
                            tensors.insert(format!("text_model.encoder.layers.{}.layer_norm1.bias", block_idx), tensor.clone());
                            tensors.insert(format!("encoder.layers.{}.layer_norm1.bias", block_idx), tensor.clone());
                        } else if subkey == "ln_2.weight" {
                            tensors.insert(format!("text_model.encoder.layers.{}.layer_norm2.weight", block_idx), tensor.clone());
                            tensors.insert(format!("encoder.layers.{}.layer_norm2.weight", block_idx), tensor.clone());
                        } else if subkey == "ln_2.bias" {
                            tensors.insert(format!("text_model.encoder.layers.{}.layer_norm2.bias", block_idx), tensor.clone());
                            tensors.insert(format!("encoder.layers.{}.layer_norm2.bias", block_idx), tensor.clone());
                        } else if subkey == "mlp.c_fc.weight" {
                            let final_tensor = self.apply_delta_if_present("te2", &format!("text_model.encoder.layers.{}.mlp.fc1.weight", block_idx), tensor, target_device, target_dtype)?;
                            tensors.insert(format!("text_model.encoder.layers.{}.mlp.fc1.weight", block_idx), final_tensor.clone());
                            tensors.insert(format!("encoder.layers.{}.mlp.fc1.weight", block_idx), final_tensor);
                        } else if subkey == "mlp.c_fc.bias" {
                            tensors.insert(format!("text_model.encoder.layers.{}.mlp.fc1.bias", block_idx), tensor.clone());
                            tensors.insert(format!("encoder.layers.{}.mlp.fc1.bias", block_idx), tensor.clone());
                        } else if subkey == "mlp.c_proj.weight" {
                            let final_tensor = self.apply_delta_if_present("te2", &format!("text_model.encoder.layers.{}.mlp.fc2.weight", block_idx), tensor, target_device, target_dtype)?;
                            tensors.insert(format!("text_model.encoder.layers.{}.mlp.fc2.weight", block_idx), final_tensor.clone());
                            tensors.insert(format!("encoder.layers.{}.mlp.fc2.weight", block_idx), final_tensor);
                        } else if subkey == "mlp.c_proj.bias" {
                            tensors.insert(format!("text_model.encoder.layers.{}.mlp.fc2.bias", block_idx), tensor.clone());
                            tensors.insert(format!("encoder.layers.{}.mlp.fc2.bias", block_idx), tensor.clone());
                        }
                    }
                }
            }
        }

        if tensors.is_empty() {
            return Err(LuminaError::MissingWeight("No OpenCLIP-G weights found in checkpoint".to_string()));
        }

        Ok(VarBuilder::from_tensors(tensors, target_dtype, target_device))
    }
}

fn translate_compvis_unet_key(key: &str) -> String {
    let mut mapped = key.to_string();

    if mapped == "input_blocks.0.0.weight" { return "conv_in.weight".to_string(); }
    if mapped == "input_blocks.0.0.bias" { return "conv_in.bias".to_string(); }
    if mapped == "out.0.weight" { return "conv_norm_out.weight".to_string(); }
    if mapped == "out.0.bias" { return "conv_norm_out.bias".to_string(); }
    if mapped == "out.2.weight" { return "conv_out.weight".to_string(); }
    if mapped == "out.2.bias" { return "conv_out.bias".to_string(); }
    if mapped == "time_embed.0.weight" { return "time_embedding.linear_1.weight".to_string(); }
    if mapped == "time_embed.0.bias" { return "time_embedding.linear_1.bias".to_string(); }
    if mapped == "time_embed.2.weight" { return "time_embedding.linear_2.weight".to_string(); }
    if mapped == "time_embed.2.bias" { return "time_embedding.linear_2.bias".to_string(); }
    if mapped == "label_emb.0.0.weight" { return "add_embedding.linear_1.weight".to_string(); }
    if mapped == "label_emb.0.0.bias" { return "add_embedding.linear_1.bias".to_string(); }
    if mapped == "label_emb.0.2.weight" { return "add_embedding.linear_2.weight".to_string(); }
    if mapped == "label_emb.0.2.bias" { return "add_embedding.linear_2.bias".to_string(); }

    // Down blocks
    mapped = mapped.replace("input_blocks.1.0.", "down_blocks.0.resnets.0.");
    mapped = mapped.replace("input_blocks.2.0.", "down_blocks.0.resnets.1.");
    mapped = mapped.replace("input_blocks.3.0.op.", "down_blocks.0.downsamplers.0.conv.");
    mapped = mapped.replace("input_blocks.3.0.", "down_blocks.0.downsamplers.0.conv.");

    mapped = mapped.replace("input_blocks.4.0.", "down_blocks.1.resnets.0.");
    mapped = mapped.replace("input_blocks.4.1.", "down_blocks.1.attentions.0.");
    mapped = mapped.replace("input_blocks.5.0.", "down_blocks.1.resnets.1.");
    mapped = mapped.replace("input_blocks.5.1.", "down_blocks.1.attentions.1.");
    mapped = mapped.replace("input_blocks.6.0.op.", "down_blocks.1.downsamplers.0.conv.");
    mapped = mapped.replace("input_blocks.6.0.", "down_blocks.1.downsamplers.0.conv.");

    mapped = mapped.replace("input_blocks.7.0.", "down_blocks.2.resnets.0.");
    mapped = mapped.replace("input_blocks.7.1.", "down_blocks.2.attentions.0.");
    mapped = mapped.replace("input_blocks.8.0.", "down_blocks.2.resnets.1.");
    mapped = mapped.replace("input_blocks.8.1.", "down_blocks.2.attentions.1.");

    // Mid block
    mapped = mapped.replace("middle_block.0.", "mid_block.resnets.0.");
    mapped = mapped.replace("middle_block.1.", "mid_block.attentions.0.");
    mapped = mapped.replace("middle_block.2.", "mid_block.resnets.1.");

    // Up blocks
    mapped = mapped.replace("output_blocks.0.0.", "up_blocks.0.resnets.0.");
    mapped = mapped.replace("output_blocks.0.1.", "up_blocks.0.attentions.0.");
    mapped = mapped.replace("output_blocks.1.0.", "up_blocks.0.resnets.1.");
    mapped = mapped.replace("output_blocks.1.1.", "up_blocks.0.attentions.1.");
    mapped = mapped.replace("output_blocks.2.0.", "up_blocks.0.resnets.2.");
    mapped = mapped.replace("output_blocks.2.1.", "up_blocks.0.attentions.2.");
    mapped = mapped.replace("output_blocks.2.2.conv.", "up_blocks.0.upsamplers.0.conv.");
    mapped = mapped.replace("output_blocks.2.2.", "up_blocks.0.upsamplers.0.conv.");

    mapped = mapped.replace("output_blocks.3.0.", "up_blocks.1.resnets.0.");
    mapped = mapped.replace("output_blocks.3.1.", "up_blocks.1.attentions.0.");
    mapped = mapped.replace("output_blocks.4.0.", "up_blocks.1.resnets.1.");
    mapped = mapped.replace("output_blocks.4.1.", "up_blocks.1.attentions.1.");
    mapped = mapped.replace("output_blocks.5.0.", "up_blocks.1.resnets.2.");
    mapped = mapped.replace("output_blocks.5.1.", "up_blocks.1.attentions.2.");
    mapped = mapped.replace("output_blocks.5.2.conv.", "up_blocks.1.upsamplers.0.conv.");
    mapped = mapped.replace("output_blocks.5.2.", "up_blocks.1.upsamplers.0.conv.");

    mapped = mapped.replace("output_blocks.6.0.", "up_blocks.2.resnets.0.");
    mapped = mapped.replace("output_blocks.7.0.", "up_blocks.2.resnets.1.");
    mapped = mapped.replace("output_blocks.8.0.", "up_blocks.2.resnets.2.");

    // Resnet internal layers
    mapped = mapped.replace(".in_layers.0.", ".norm1.");
    mapped = mapped.replace(".in_layers.2.", ".conv1.");
    mapped = mapped.replace(".out_layers.0.", ".norm2.");
    mapped = mapped.replace(".out_layers.3.", ".conv2.");
    mapped = mapped.replace(".emb_layers.1.", ".time_emb_proj.");
    mapped = mapped.replace(".skip_connection.", ".conv_shortcut.");

    mapped
}

fn translate_compvis_vae_key(key: &str) -> String {
    let mut mapped = key.to_string();

    // Mid attention mappings first
    mapped = mapped.replace("encoder.mid.attn_1.norm.", "encoder.mid_block.attentions.0.group_norm.");
    mapped = mapped.replace("encoder.mid.attn_1.q.", "encoder.mid_block.attentions.0.to_q.");
    mapped = mapped.replace("encoder.mid.attn_1.k.", "encoder.mid_block.attentions.0.to_k.");
    mapped = mapped.replace("encoder.mid.attn_1.v.", "encoder.mid_block.attentions.0.to_v.");
    mapped = mapped.replace("encoder.mid.attn_1.proj_out.", "encoder.mid_block.attentions.0.to_out.0.");

    mapped = mapped.replace("decoder.mid.attn_1.norm.", "decoder.mid_block.attentions.0.group_norm.");
    mapped = mapped.replace("decoder.mid.attn_1.q.", "decoder.mid_block.attentions.0.to_q.");
    mapped = mapped.replace("decoder.mid.attn_1.k.", "decoder.mid_block.attentions.0.to_k.");
    mapped = mapped.replace("decoder.mid.attn_1.v.", "decoder.mid_block.attentions.0.to_v.");
    mapped = mapped.replace("decoder.mid.attn_1.proj_out.", "decoder.mid_block.attentions.0.to_out.0.");

    // Encoder down blocks
    for i in 0..4 {
        for j in 0..3 {
            mapped = mapped.replace(
                &format!("encoder.down.{}.block.{}.", i, j),
                &format!("encoder.down_blocks.{}.resnets.{}.", i, j),
            );
        }
        mapped = mapped.replace(
            &format!("encoder.down.{}.downsample.conv.", i),
            &format!("encoder.down_blocks.{}.downsamplers.0.conv.", i),
        );
    }

    // Decoder up blocks (CompVis decoder.up.0 is Diffusers up_blocks.3)
    for i in 0..4 {
        let diffusers_up_idx = 3 - i;
        for j in 0..3 {
            mapped = mapped.replace(
                &format!("decoder.up.{}.block.{}.", i, j),
                &format!("decoder.up_blocks.{}.resnets.{}.", diffusers_up_idx, j),
            );
        }
        mapped = mapped.replace(
            &format!("decoder.up.{}.upsample.conv.", i),
            &format!("decoder.up_blocks.{}.upsamplers.0.conv.", diffusers_up_idx),
        );
    }

    // Mid block Resnets
    mapped = mapped.replace("encoder.mid.block_1.", "encoder.mid_block.resnets.0.");
    mapped = mapped.replace("encoder.mid.block_2.", "encoder.mid_block.resnets.1.");
    mapped = mapped.replace("decoder.mid.block_1.", "decoder.mid_block.resnets.0.");
    mapped = mapped.replace("decoder.mid.block_2.", "decoder.mid_block.resnets.1.");

    // Norm out
    mapped = mapped.replace(".norm_out.", ".conv_norm_out.");
    mapped = mapped.replace(".nin_shortcut.", ".conv_shortcut.");

    mapped
}

impl WeightsSource for SafeTensorsArchive {
    fn get_tensor(&self, name: &str, device: &Device, dtype: DType) -> Result<Tensor> {
        SafeTensorsArchive::get_tensor(self, name, device, dtype)
    }

    fn contains(&self, name: &str) -> bool {
        SafeTensorsArchive::contains(self, name)
    }

    fn raw_info(&self, name: &str) -> Option<(safetensors::Dtype, Vec<usize>)> {
        SafeTensorsArchive::raw_info(self, name)
    }

    fn keys(&self) -> Vec<String> {
        self.tensors.keys().cloned().collect()
    }

    fn describe(&self) -> String {
        format!("safetensors ({} shards)", self._mmaps.len())
    }
}
