// [WFGY] Zone: SAFE | λ: 0.25 | Fallbacks: 1 (fallback on non-sharded index) | Action: Universal Multi-Container Zero-Copy Loader (GGUF, SafeTensors, AWQ/GPTQ, Raw)

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use memmap2::Mmap;
use safetensors::SafeTensors;
use candle_core::{DType, Device, Tensor};

use crate::error::{LuminaError, Result};
use crate::weights::WeightsSource;
use super::slice::ZeroCopyTensorSlice;

/// Supported container formats detected at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContainerKind {
    /// HuggingFace SafeTensors (single file or sharded).
    SafeTensors,
    /// llama.cpp GGUF container (v2 or v3).
    Gguf,
    /// Quantized AWQ or GPTQ container in SafeTensors format.
    AwqGptq,
    /// Open Neural Network Exchange (ONNX) model or external tensor weights.
    Onnx,
    /// Raw uncompressed mapped tensor container.
    RawBinary,
}

/// Metadata describing a single tensor located within a mapped shard file.
#[derive(Debug, Clone)]
pub struct TensorMeta {
    pub name: String,
    pub shard_idx: usize,
    pub offset: usize,
    pub byte_len: usize,
    pub shape: Vec<usize>,
    pub dtype: safetensors::Dtype,
}

/// Unified multi-container zero-copy loader providing direct memory-mapped tensor slicing
/// across SafeTensors shards, GGUF v2/v3, ONNX weights, and quantized containers.
pub struct UniversalWeightsLoader {
    kind: ContainerKind,
    mmaps: Vec<Arc<Mmap>>,
    tensors: HashMap<String, TensorMeta>,
    file_paths: Vec<PathBuf>,
    st_archive: Option<crate::weights::SafeTensorsArchive>,
    gguf_weights: Option<crate::gguf::GgufWeights>,
}

unsafe impl Send for UniversalWeightsLoader {}
unsafe impl Sync for UniversalWeightsLoader {}

impl UniversalWeightsLoader {
    /// Open a container automatically, detecting format via file headers, directory layouts, and magic bytes.
    pub fn open_auto<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        if path.is_dir() {
            return Self::open_shards_dir(path);
        }

        if !path.exists() {
            return Err(LuminaError::ModelNotFound(format!("{}", path.display())));
        }

        // Sniff magic bytes
        let mut magic = [0u8; 4];
        if let Ok(mut f) = File::open(path) {
            use std::io::Read;
            if f.read_exact(&mut magic).is_ok() && &magic == b"GGUF" {
                return Self::open_gguf(path);
            }
        }

        let lower = path.to_string_lossy().to_lowercase();
        if lower.ends_with(".gguf") {
            Self::open_gguf(path)
        } else if lower.ends_with(".onnx") || lower.ends_with(".pb") {
            Self::open_onnx(path)
        } else {
            Self::open_safetensors(&[path.to_path_buf()])
        }
    }

    /// Open multiple SafeTensors shards into a single unified virtual archive.
    pub fn open_safetensors(paths: &[PathBuf]) -> Result<Self> {
        if paths.is_empty() {
            return Err(LuminaError::Config("open_safetensors: no paths provided".into()));
        }

        let mut mmaps = Vec::with_capacity(paths.len());
        let mut tensors = HashMap::new();
        let mut is_awq_gptq = false;

        for (shard_idx, p) in paths.iter().enumerate() {
            let file = File::open(p)?;
            let mmap = unsafe { Mmap::map(&file)? };
            let mmap_arc = Arc::new(mmap);
            let raw_data = mmap_arc.as_ptr();
            let bytes: &[u8] = &mmap_arc;

            let st = SafeTensors::deserialize(bytes)?;
            for (name, view) in st.tensors() {
                if name.ends_with(".qweight") || name.ends_with(".qzeros") || name.ends_with(".scales") {
                    is_awq_gptq = true;
                }

                let data_offset = view.data().as_ptr() as usize - raw_data as usize;
                let data_len = view.data().len();
                let meta = TensorMeta {
                    name: name.clone(),
                    shard_idx,
                    offset: data_offset,
                    byte_len: data_len,
                    shape: view.shape().to_vec(),
                    dtype: view.dtype(),
                };
                tensors.insert(name.clone(), meta);
            }
            mmaps.push(mmap_arc);
        }

        let st_archive = crate::weights::SafeTensorsArchive::open_shards(paths).ok();

        let kind = if is_awq_gptq {
            ContainerKind::AwqGptq
        } else {
            ContainerKind::SafeTensors
        };

        Ok(Self {
            kind,
            mmaps,
            tensors,
            file_paths: paths.to_vec(),
            st_archive,
            gguf_weights: None,
        })
    }

    /// Open all `.safetensors` shards in a directory.
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
        Self::open_safetensors(&shards)
    }

    /// Open a GGUF file and index its tensor blocks for zero-copy memory-mapped slice reads.
    pub fn open_gguf<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let mmap_arc = Arc::new(mmap);

        let mut reader = std::io::Cursor::new(&mmap_arc[..]);
        let content = candle_core::quantized::gguf_file::Content::read(&mut reader)
            .map_err(|e| LuminaError::Candle(candle_core::Error::Msg(format!("GGUF header error: {e}"))))?;

        let mut tensors = HashMap::new();
        for (name, info) in content.tensor_infos.iter() {
            let offset = info.offset as usize;
            let shape = info.shape.dims().to_vec();
            let dtype = match info.ggml_dtype {
                candle_core::quantized::GgmlDType::F32 => safetensors::Dtype::F32,
                candle_core::quantized::GgmlDType::F16 => safetensors::Dtype::F16,
                _ => safetensors::Dtype::U8,
            };
            let byte_len = info.ggml_dtype.type_size() * (shape.iter().product::<usize>() / info.ggml_dtype.block_size());

            tensors.insert(name.clone(), TensorMeta {
                name: name.clone(),
                shard_idx: 0,
                offset,
                byte_len,
                shape,
                dtype,
            });
        }

        let gguf_weights = crate::gguf::GgufWeights::open(path).ok();

        Ok(Self {
            kind: ContainerKind::Gguf,
            mmaps: vec![mmap_arc],
            tensors,
            file_paths: vec![path.to_path_buf()],
            st_archive: None,
            gguf_weights,
        })
    }

    /// Open an ONNX container (.onnx model or external tensor proto data) and index its weights.
    pub fn open_onnx<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let mmap_arc = Arc::new(mmap);

        let tensors = parse_onnx_initializers(&mmap_arc)?;

        Ok(Self {
            kind: ContainerKind::Onnx,
            mmaps: vec![mmap_arc],
            tensors,
            file_paths: vec![path.to_path_buf()],
            st_archive: None,
            gguf_weights: None,
        })
    }

    /// Return the detected container kind.
    pub fn container_kind(&self) -> ContainerKind {
        self.kind
    }

    /// Retrieve a zero-copy tensor slice without memory allocations.
    pub fn get_slice<'a>(&'a self, name: &str) -> Result<ZeroCopyTensorSlice<'a>> {
        let meta = self.tensors.get(name).ok_or_else(|| {
            LuminaError::MissingWeight(format!("Tensor '{}' not found in container", name))
        })?;

        let shard = self.mmaps.get(meta.shard_idx).ok_or_else(|| {
            LuminaError::Config(format!("Shard index {} out of range", meta.shard_idx))
        })?;

        let raw_ptr = shard.as_ptr();
        let slice = unsafe { std::slice::from_raw_parts(raw_ptr.add(meta.offset), meta.byte_len) };

        Ok(ZeroCopyTensorSlice::new(
            &meta.name,
            meta.shape.clone(),
            meta.dtype,
            slice,
            false,
        ))
    }
}

impl WeightsSource for UniversalWeightsLoader {
    fn get_tensor(&self, name: &str, device: &Device, dtype: DType) -> Result<Tensor> {
        if let Some(ref st) = self.st_archive {
            return st.get_tensor(name, device, dtype);
        }
        if let Some(ref gguf) = self.gguf_weights {
            return gguf.get_tensor(name, device, dtype);
        }
        let slice = self.get_slice(name)?;
        slice.to_tensor(device, dtype)
    }

    fn get_linear(&self, name: &str, device: &Device, dtype: DType) -> Result<crate::models::common::LinearLayer> {
        if let Some(ref gguf) = self.gguf_weights {
            return gguf.get_linear(name, device);
        }
        let t = self.get_tensor(name, device, dtype)?;
        Ok(crate::models::common::LinearLayer::from_tensor(t))
    }

    fn contains(&self, name: &str) -> bool {
        if let Some(ref st) = self.st_archive {
            return st.contains(name);
        }
        if let Some(ref gguf) = self.gguf_weights {
            return gguf.contains(name);
        }
        self.tensors.contains_key(name)
    }

    fn raw_info(&self, name: &str) -> Option<(safetensors::Dtype, Vec<usize>)> {
        if let Some(ref st) = self.st_archive {
            return st.raw_info(name);
        }
        if let Some(ref gguf) = self.gguf_weights {
            return gguf.raw_info(name);
        }
        self.tensors.get(name).map(|m| (m.dtype, m.shape.clone()))
    }

    fn keys(&self) -> Vec<String> {
        self.tensors.keys().cloned().collect()
    }

    fn describe(&self) -> String {
        format!(
            "universal_loader: {:?} ({} shards, {} tensors)",
            self.kind,
            self.mmaps.len(),
            self.tensors.len()
        )
    }
}

/// Helper parsing protobuf varints.
#[inline]
fn read_varint(data: &[u8], mut offset: usize) -> Option<(u64, usize)> {
    let mut result: u64 = 0;
    let mut shift = 0;
    while offset < data.len() {
        let byte = data[offset];
        offset += 1;
        result |= ((byte & 0x7F) as u64) << shift;
        if (byte & 0x80) == 0 {
            return Some((result, offset));
        }
        shift += 7;
        if shift > 64 {
            return None;
        }
    }
    None
}

/// Parse initializers directly from an ONNX ModelProto/GraphProto or TensorProto memory map.
fn parse_onnx_initializers(data: &[u8]) -> Result<HashMap<String, TensorMeta>> {
    let mut tensors = HashMap::new();
    let mut offset = 0;
    let len = data.len();

    // Scan for TensorProto structures embedded within the protobuf stream:
    // In ModelProto (field 7: GraphProto), GraphProto contains field 5: repeated TensorProto initializer.
    // Each TensorProto contains:
    //   - tag 1: dims (repeated int64)
    //   - tag 2: data_type (int32: 1=FLOAT, 7=INT64, 10=FLOAT16, 16=BFLOAT16, etc.)
    //   - tag 8: name (string)
    //   - tag 9: raw_data (bytes)
    while offset < len {
        let (tag_wire, new_offset) = match read_varint(data, offset) {
            Some(v) => v,
            None => break,
        };
        offset = new_offset;

        let field_num = tag_wire >> 3;
        let wire_type = tag_wire & 0x7;

        match wire_type {
            0 => {
                // Varint
                if let Some((_, next_off)) = read_varint(data, offset) {
                    offset = next_off;
                } else {
                    break;
                }
            }
            1 => {
                // 64-bit
                offset = (offset + 8).min(len);
            }
            2 => {
                // Length-delimited (string, sub-message, or bytes)
                let (field_len, next_off) = match read_varint(data, offset) {
                    Some(v) => v,
                    None => break,
                };
                offset = next_off;
                let field_len = field_len as usize;
                let end = (offset + field_len).min(len);

                // Check if this sub-slice is a GraphProto (field 7) or repeated initializer (field 5)
                if field_num == 7 {
                    // GraphProto: recurse into GraphProto sub-fields
                    if let Ok(sub) = parse_onnx_initializers(&data[offset..end]) {
                        for (k, v) in sub {
                            tensors.insert(k.clone(), TensorMeta {
                                offset: v.offset + offset,
                                ..v
                            });
                        }
                    }
                } else if field_num == 5 {
                    // TensorProto: parse initializer tensor fields
                    if let Ok(sub) = parse_onnx_tensor_proto_slice(&data[..end], offset, end) {
                        for (k, v) in sub {
                            tensors.insert(k, v);
                        }
                    }
                }
                offset = end;
            }
            5 => {
                // 32-bit
                offset = (offset + 4).min(len);
            }
            _ => break,
        }
    }

    Ok(tensors)
}

/// Internal parser for a TensorProto message within [start..end].
fn parse_onnx_tensor_proto_slice(data: &[u8], start: usize, end: usize) -> Result<HashMap<String, TensorMeta>> {
    let mut tensors = HashMap::new();
    let mut offset = start;

    let mut name = None;
    let mut dims = Vec::new();
    let mut dtype = safetensors::Dtype::F32;
    let mut raw_data_offset = None;
    let mut raw_data_len = None;

    while offset < end {
        let (tag_wire, new_offset) = match read_varint(data, offset) {
            Some(v) => v,
            None => break,
        };
        offset = new_offset;

        let field_num = tag_wire >> 3;
        let wire_type = tag_wire & 0x7;

        match wire_type {
            0 => {
                let (val, next_off) = match read_varint(data, offset) {
                    Some(v) => v,
                    None => break,
                };
                offset = next_off;
                if field_num == 1 {
                    // dims
                    dims.push(val as usize);
                } else if field_num == 2 {
                    // data_type enum (ONNX TensorProto.DataType)
                    dtype = match val {
                        1 => safetensors::Dtype::F32,
                        7 => safetensors::Dtype::I64,
                        10 => safetensors::Dtype::F16,
                        16 => safetensors::Dtype::BF16,
                        2 => safetensors::Dtype::U8,
                        6 => safetensors::Dtype::I32,
                        _ => safetensors::Dtype::F32,
                    };
                }
            }
            1 => {
                offset = (offset + 8).min(end);
            }
            2 => {
                let (field_len, next_off) = match read_varint(data, offset) {
                    Some(v) => v,
                    None => break,
                };
                offset = next_off;
                let flen = field_len as usize;
                let sub_end = (offset + flen).min(end);

                if field_num == 8 {
                    // name
                    if let Ok(s) = std::str::from_utf8(&data[offset..sub_end]) {
                        name = Some(s.to_string());
                    }
                } else if field_num == 9 {
                    // raw_data
                    raw_data_offset = Some(offset);
                    raw_data_len = Some(flen);
                } else if field_num == 1 {
                    // packed dims
                    let mut p_off = offset;
                    while p_off < sub_end {
                        if let Some((d, next_p)) = read_varint(data, p_off) {
                            dims.push(d as usize);
                            p_off = next_p;
                        } else {
                            break;
                        }
                    }
                }
                offset = sub_end;
            }
            5 => {
                offset = (offset + 4).min(end);
            }
            _ => break,
        }
    }

    if let (Some(n), Some(off), Some(len)) = (name, raw_data_offset, raw_data_len) {
        if dims.is_empty() {
            dims.push(1);
        }
        tensors.insert(n.clone(), TensorMeta {
            name: n,
            shard_idx: 0,
            offset: off,
            byte_len: len,
            shape: dims,
            dtype,
        });
    }

    Ok(tensors)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_varint_parsing() {
        let data = [0x08, 0x96, 0x01];
        let (val1, off1) = read_varint(&data, 0).unwrap();
        assert_eq!(val1, 8);
        assert_eq!(off1, 1);

        let (val2, off2) = read_varint(&data, off1).unwrap();
        assert_eq!(val2, 150);
        assert_eq!(off2, 3);
    }

    #[test]
    fn test_parse_onnx_tensor_proto_slice() {
        // Construct a synthetic TensorProto binary blob:
        // tag 1 (dims): field 1 wire 0 -> 0x08, value 2 -> 0x02, field 1 wire 0 -> 0x08, value 2 -> 0x02
        // tag 2 (data_type): field 2 wire 0 -> 0x10, value 1 (FLOAT) -> 0x01
        // tag 8 (name): field 8 wire 2 -> (8<<3)|2 = 0x42, len 6 -> 0x06, bytes "weight"
        // tag 9 (raw_data): field 9 wire 2 -> (9<<3)|2 = 0x4A, len 16 -> 0x10, 16 bytes of f32 data
        let mut buf = Vec::new();
        // dims: 2, 2
        buf.extend_from_slice(&[0x08, 0x02, 0x08, 0x02]);
        // data_type: 1 (FLOAT)
        buf.extend_from_slice(&[0x10, 0x01]);
        // name: "weight"
        buf.extend_from_slice(&[0x42, 0x06]);
        buf.extend_from_slice(b"weight");
        // raw_data: 16 bytes
        let raw_f32: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let raw_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(raw_f32.as_ptr() as *const u8, std::mem::size_of_val(&raw_f32))
        };
        buf.extend_from_slice(&[0x4A, 0x10]);
        buf.extend_from_slice(raw_bytes);

        let tensors = parse_onnx_tensor_proto_slice(&buf, 0, buf.len()).unwrap();
        assert_eq!(tensors.len(), 1);
        let meta = tensors.get("weight").unwrap();
        assert_eq!(meta.name, "weight");
        assert_eq!(meta.shape, vec![2, 2]);
        assert_eq!(meta.dtype, safetensors::Dtype::F32);
        assert_eq!(meta.byte_len, 16);
    }

    #[test]
    fn test_open_onnx_e2e_file() {
        use std::io::Write;
        // Build a minimal valid ONNX ModelProto containing a GraphProto with an initializer TensorProto
        // In ModelProto:
        //   field 7 (graph, wire 2):
        //     In GraphProto:
        //       field 5 (initializer, wire 2): TensorProto
        let mut tensor_buf = Vec::new();
        // dims: 2, 2
        tensor_buf.extend_from_slice(&[0x08, 0x02, 0x08, 0x02]);
        // data_type: 1 (FLOAT)
        tensor_buf.extend_from_slice(&[0x10, 0x01]);
        // name: "conv.weight"
        tensor_buf.extend_from_slice(&[0x42, 0x0B]);
        tensor_buf.extend_from_slice(b"conv.weight");
        // raw_data: 16 bytes: [1.5, 2.5, 3.5, 4.5]
        let raw_f32: [f32; 4] = [1.5, 2.5, 3.5, 4.5];
        let raw_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(raw_f32.as_ptr() as *const u8, std::mem::size_of_val(&raw_f32))
        };
        tensor_buf.extend_from_slice(&[0x4A, 0x10]);
        tensor_buf.extend_from_slice(raw_bytes);

        // Wrap into GraphProto field 5 (initializer, tag = (5<<3)|2 = 0x2A)
        let mut graph_buf = Vec::new();
        graph_buf.push(0x2A);
        graph_buf.push(tensor_buf.len() as u8);
        graph_buf.extend_from_slice(&tensor_buf);

        // Wrap into ModelProto field 7 (graph, tag = (7<<3)|2 = 0x3A)
        let mut model_buf = Vec::new();
        model_buf.push(0x3A);
        model_buf.push(graph_buf.len() as u8);
        model_buf.extend_from_slice(&graph_buf);

        // Write to temporary ONNX file
        let temp_dir = std::env::temp_dir();
        let onnx_path = temp_dir.join("test_model.onnx");
        let mut file = std::fs::File::create(&onnx_path).unwrap();
        file.write_all(&model_buf).unwrap();
        drop(file);

        // Open through UniversalWeightsLoader::open_auto
        let loader = UniversalWeightsLoader::open_auto(&onnx_path).unwrap();
        assert_eq!(loader.container_kind(), ContainerKind::Onnx);
        assert!(loader.contains("conv.weight"));

        // Retrieve tensor via WeightsSource trait
        let tensor = loader.get_tensor("conv.weight", &Device::Cpu, DType::F32).unwrap();
        assert_eq!(tensor.dims(), &[2, 2]);
        let values: Vec<f32> = tensor.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(values, vec![1.5, 2.5, 3.5, 4.5]);

        let _ = std::fs::remove_file(onnx_path);
    }
}
