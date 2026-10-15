// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: Zero-Copy Memory-Mapped Tensor Slice Abstraction

use candle_core::{DType, Device, Tensor};
use half::{bf16, f16};
use crate::error::{LuminaError, Result};

#[inline]
fn cast_slice<T: Copy>(bytes: &[u8]) -> std::borrow::Cow<'_, [T]> {
    let elem_size = std::mem::size_of::<T>();
    let align = std::mem::align_of::<T>();
    if elem_size == 0 {
        return std::borrow::Cow::Borrowed(&[]);
    }
    let count = bytes.len() / elem_size;
    let ptr = bytes.as_ptr();
    if (ptr as usize) % align == 0 {
        std::borrow::Cow::Borrowed(unsafe { std::slice::from_raw_parts(ptr as *const T, count) })
    } else {
        // Unaligned memory slice (e.g. arbitrary protobuf offset): copy into aligned buffer
        let mut aligned_vec = Vec::with_capacity(count);
        unsafe {
            aligned_vec.set_len(count);
            std::ptr::copy_nonoverlapping(ptr, aligned_vec.as_mut_ptr() as *mut u8, count * elem_size);
        }
        std::borrow::Cow::Owned(aligned_vec)
    }
}

#[inline]
#[allow(dead_code)]
fn cast_to_bytes<T>(slice: &[T]) -> &[u8] {
    let elem_size = std::mem::size_of::<T>();
    let byte_count = slice.len() * elem_size;
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, byte_count) }
}

/// A zero-copy memory-mapped tensor slice mapped directly from persistent container files
/// (SafeTensors, GGUF v2/v3, AWQ/GPTQ, or raw binary buffers).
#[derive(Debug, Clone)]
pub struct ZeroCopyTensorSlice<'a> {
    /// Canonical tensor name.
    pub name: String,
    /// Tensor dimensions.
    pub shape: Vec<usize>,
    /// Underlying storage data type.
    pub dtype: safetensors::Dtype,
    /// Direct memory-mapped byte view.
    pub data: &'a [u8],
    /// Whether the underlying memory is host page-locked (pinned DMA buffer).
    pub is_pinned: bool,
}

impl<'a> ZeroCopyTensorSlice<'a> {
    /// Create a new zero-copy tensor slice.
    pub fn new(
        name: impl Into<String>,
        shape: Vec<usize>,
        dtype: safetensors::Dtype,
        data: &'a [u8],
        is_pinned: bool,
    ) -> Self {
        Self {
            name: name.into(),
            shape,
            dtype,
            data,
            is_pinned,
        }
    }

    /// Total number of scalar elements.
    pub fn num_elements(&self) -> usize {
        if self.shape.is_empty() {
            0
        } else {
            self.shape.iter().product()
        }
    }

    /// Total byte length of the slice.
    pub fn byte_len(&self) -> usize {
        self.data.len()
    }

    /// Materialize directly into a Candle `Tensor` on the target device and dtype.
    /// When types match, utilizes direct zero-copy slice ingestion without intermediate heap copies.
    pub fn to_tensor(&self, device: &Device, target_dtype: DType) -> Result<Tensor> {
        let shape = self.shape.as_slice();

        match (self.dtype, target_dtype) {
            (safetensors::Dtype::F32, DType::F32) => {
                let slice = cast_slice::<f32>(self.data);
                Ok(Tensor::from_slice(slice.as_ref(), shape, device).map_err(LuminaError::Candle)?)
            }
            (safetensors::Dtype::F16, DType::F16) => {
                let slice = cast_slice::<f16>(self.data);
                Ok(Tensor::from_slice(slice.as_ref(), shape, device).map_err(LuminaError::Candle)?)
            }
            (safetensors::Dtype::BF16, DType::BF16) => {
                let slice = cast_slice::<bf16>(self.data);
                Ok(Tensor::from_slice(slice.as_ref(), shape, device).map_err(LuminaError::Candle)?)
            }
            (safetensors::Dtype::I64, DType::I64) => {
                let slice = cast_slice::<i64>(self.data);
                Ok(Tensor::from_slice(slice.as_ref(), shape, device).map_err(LuminaError::Candle)?)
            }
            (safetensors::Dtype::U8, DType::U8) => {
                Ok(Tensor::from_slice(self.data, shape, device).map_err(LuminaError::Candle)?)
            }
            (safetensors::Dtype::F16, DType::F32) => {
                let slice = cast_slice::<f16>(self.data);
                let f32_vec: Vec<f32> = slice.iter().map(|&x| x.to_f32()).collect();
                Ok(Tensor::from_vec(f32_vec, shape, device).map_err(LuminaError::Candle)?)
            }
            (safetensors::Dtype::BF16, DType::F32) => {
                let slice = cast_slice::<bf16>(self.data);
                let f32_vec: Vec<f32> = slice.iter().map(|&x| x.to_f32()).collect();
                Ok(Tensor::from_vec(f32_vec, shape, device).map_err(LuminaError::Candle)?)
            }
            (safetensors::Dtype::F32, DType::F16) => {
                let slice = cast_slice::<f32>(self.data);
                let f16_vec: Vec<f16> = slice.iter().map(|&x| f16::from_f32(x)).collect();
                Ok(Tensor::from_vec(f16_vec, shape, device).map_err(LuminaError::Candle)?)
            }
            (safetensors::Dtype::F32, DType::BF16) => {
                let slice = cast_slice::<f32>(self.data);
                let bf16_vec: Vec<bf16> = slice.iter().map(|&x| bf16::from_f32(x)).collect();
                Ok(Tensor::from_vec(bf16_vec, shape, device).map_err(LuminaError::Candle)?)
            }
            (safetensors::Dtype::BF16, DType::F16) => {
                let slice = cast_slice::<bf16>(self.data);
                let f16_vec: Vec<f16> = slice.iter().map(|&x| f16::from_f32(x.to_f32())).collect();
                Ok(Tensor::from_vec(f16_vec, shape, device).map_err(LuminaError::Candle)?)
            }
            (safetensors::Dtype::F16, DType::BF16) => {
                let slice = cast_slice::<f16>(self.data);
                let bf16_vec: Vec<bf16> = slice.iter().map(|&x| bf16::from_f32(x.to_f32())).collect();
                Ok(Tensor::from_vec(bf16_vec, shape, device).map_err(LuminaError::Candle)?)
            }
            (safetensors::Dtype::I32, DType::I64) => {
                let slice = cast_slice::<i32>(self.data);
                let i64_vec: Vec<i64> = slice.iter().map(|&x| x as i64).collect();
                Ok(Tensor::from_vec(i64_vec, shape, device).map_err(LuminaError::Candle)?)
            }
            _ => {
                // Generic fallback via F32 intermediate
                let f32_tensor = match self.dtype {
                    safetensors::Dtype::F32 => {
                        let slice = cast_slice::<f32>(self.data);
                        Tensor::from_slice(slice.as_ref(), shape, &Device::Cpu).map_err(LuminaError::Candle)?
                    }
                    safetensors::Dtype::F16 => {
                        let slice = cast_slice::<f16>(self.data);
                        let v: Vec<f32> = slice.iter().map(|&x| x.to_f32()).collect();
                        Tensor::from_vec(v, shape, &Device::Cpu).map_err(LuminaError::Candle)?
                    }
                    safetensors::Dtype::BF16 => {
                        let slice = cast_slice::<bf16>(self.data);
                        let v: Vec<f32> = slice.iter().map(|&x| x.to_f32()).collect();
                        Tensor::from_vec(v, shape, &Device::Cpu).map_err(LuminaError::Candle)?
                    }
                    safetensors::Dtype::I32 => {
                        let slice = cast_slice::<i32>(self.data);
                        let v: Vec<f32> = slice.iter().map(|&x| x as f32).collect();
                        Tensor::from_vec(v, shape, &Device::Cpu).map_err(LuminaError::Candle)?
                    }
                    other => {
                        return Err(LuminaError::Config(format!(
                            "Unsupported tensor slice dtype conversion: {:?} -> {:?}",
                            other, target_dtype
                        )));
                    }
                };

                let converted = f32_tensor.to_dtype(target_dtype).map_err(LuminaError::Candle)?;
                if device.is_cpu() {
                    Ok(converted)
                } else {
                    Ok(converted.to_device(device).map_err(LuminaError::Candle)?)
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_copy_tensor_slice_conversion() {
        let f32_data: [f32; 4] = [1.5, -2.5, 3.0, 0.0];
        let bytes: &[u8] = cast_to_bytes(&f32_data);

        let slice = ZeroCopyTensorSlice::new(
            "test_tensor",
            vec![2, 2],
            safetensors::Dtype::F32,
            bytes,
            false,
        );

        assert_eq!(slice.num_elements(), 4);
        assert_eq!(slice.byte_len(), 16);

        let tensor = slice.to_tensor(&Device::Cpu, DType::F32).unwrap();
        assert_eq!(tensor.dims(), &[2, 2]);
        let extracted: Vec<f32> = tensor.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(extracted, vec![1.5, -2.5, 3.0, 0.0]);

        // Convert to F16 target dtype
        let t_f16 = slice.to_tensor(&Device::Cpu, DType::F16).unwrap();
        assert_eq!(t_f16.dtype(), DType::F16);
    }
}
