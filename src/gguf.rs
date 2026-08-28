// [WFGY] Zone: SAFE | λ: 0.35 | Fallbacks: 0 | Action: GGUF (llama.cpp) weight source — quantized weight access via a unified WeightsSource brick

use crate::error::{LuminaError, Result};
use crate::weights::WeightsSource;
use candle_core::quantized::{gguf_file, QTensor};
use candle_core::{DType, Device, Tensor};
use std::fs::File;
use std::path::Path;

/// A [`WeightsSource`] backed by a single GGUF (llama.cpp) file.
///
/// GGUF stores tensors quantized (Q4_K, Q5, FP8, ...). This brick exposes them through the same
/// trait as [`SafeTensorsArchive`], so encoders / transformers can be assembled from a GGUF model
/// by name, keyed exactly like the GGUF metadata (`model.layers.0.self_attn.…`). Every `get_tensor`
/// call dequantizes the corresponding QTensor onto `device` and (if needed) converts to the requested
/// `dtype`.
///
/// This is the companion brick for lighter community quantizations (llama.cpp / GGUF exports).
pub struct GgufWeights {
    file: File,
    content: gguf_file::Content,
}

impl GgufWeights {
    /// Open a GGUF file and read its header (tensor metadata).
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path.as_ref())?;
        let mut reader = std::io::BufReader::new(file);
        let content = gguf_file::Content::read(&mut reader)
            .map_err(|e| LuminaError::Context {
                context: format!("failed to read GGUF header: {e}"),
                source: Box::new(crate::error::LuminaError::Candle(e)),
            })?;
        let file = reader.into_inner();
        Ok(Self { file, content })
    }

    /// Whether a tensor exists under the given GGUF name.
    pub fn has(&self, name: &str) -> bool {
        self.content.tensor_infos.contains_key(name)
    }
}

impl WeightsSource for GgufWeights {
    fn get_tensor(&self, name: &str, device: &Device, dtype: DType) -> Result<Tensor> {
        let q: QTensor = self
            .content
            .tensor(&mut std::io::BufReader::new(&self.file), name, device)
            .map_err(|e| LuminaError::Context {
                context: format!("failed to decode GGUF tensor '{name}': {e}"),
                source: Box::new(crate::error::LuminaError::Candle(e)),
            })?;
        let t = q
            .dequantize(device)
            .map_err(|e| LuminaError::Context {
                context: format!("failed to dequantize GGUF tensor '{name}': {e}"),
                source: Box::new(crate::error::LuminaError::Candle(e)),
            })?;
        if t.dtype() == dtype {
            Ok(t)
        } else {
            t.to_dtype(dtype).map_err(crate::error::LuminaError::Candle)
        }
    }

    fn contains(&self, name: &str) -> bool {
        self.content.tensor_infos.contains_key(name)
    }

    fn raw_info(&self, name: &str) -> Option<(safetensors::Dtype, Vec<usize>)> {
        let info = self.content.tensor_infos.get(name)?;
        // GGUF uses GgmlDType, not safetensors Dtype. We report a best-effort equivalent so
        // architecture detection (which only needs shapes) still works.
        let st = match info.ggml_dtype {
            candle_core::quantized::GgmlDType::F32 => safetensors::Dtype::F32,
            candle_core::quantized::GgmlDType::F16 => safetensors::Dtype::F16,
            _ => safetensors::Dtype::U8,
        };
        Some((st, info.shape.dims().to_vec()))
    }

    fn keys(&self) -> Vec<String> {
        self.content.tensor_infos.keys().cloned().collect()
    }

    fn describe(&self) -> String {
        format!("gguf ({} tensors)", self.content.tensor_infos.len())
    }
}
