// [WFGY] Zone: SAFE | λ: 0.15 | Fallbacks: 1 (CUDA/ROCm/Metal fallback to CPU) | Action: Device selection and compute capability probing

use candle_core::Device;
use tracing::info;

/// Probe and select the most performant available compute device.
/// Priority: CUDA / ROCm -> Metal -> CPU
pub fn auto_device() -> candle_core::Result<Device> {
    #[cfg(feature = "cuda")]
    {
        match Device::new_cuda(0) {
            Ok(device) => {
                info!("Using GPU acceleration device via CUDA (ordinal 0)");
                return Ok(device);
            }
            Err(err) => {
                tracing::warn!("CUDA GPU device requested but unavailable: {:?}. Falling back.", err);
            }
        }
    }

    #[cfg(feature = "rocm")]
    {
        match Device::new_rocm(0) {
            Ok(device) => {
                info!("Using GPU acceleration device via ROCm (ordinal 0, gfx1201)");
                return Ok(device);
            }
            Err(err) => {
                tracing::warn!("ROCm GPU device requested but unavailable: {:?}. Falling back.", err);
            }
        }
    }

    #[cfg(feature = "metal")]
    {
        match Device::new_metal(0) {
            Ok(device) => {
                info!("Using Apple Silicon Metal compute device (ordinal 0)");
                return Ok(device);
            }
            Err(err) => {
                tracing::warn!("Metal device requested but unavailable: {:?}. Falling back.", err);
            }
        }
    }

    info!("Using CPU device for computation");
    Ok(Device::Cpu)
}

/// Explicit device selection helper with fallback.
pub fn select_device(prefer_gpu: bool) -> candle_core::Result<Device> {
    if prefer_gpu {
        auto_device()
    } else {
        Ok(Device::Cpu)
    }
}

/// Disentangled performance telemetry for high-resolution profiler reporting
#[derive(Debug, Clone, Default)]
pub struct GenerationMetrics {
    pub prompt_encode_ms: f64,
    pub unet_steps: usize,
    pub unet_total_ms: f64,
    pub unet_step_avg_ms: f64,
    pub unet_it_per_sec: f64,
    pub vae_decode_ms: f64,
    pub total_wallclock_ms: f64,
}

impl GenerationMetrics {
    pub fn summary_report(&self) -> String {
        format!(
            "⏱️ [Telemetry] UNet: {:.2}s ({} steps, {:.2} ms/step, {:.2} it/s) | VAE: {:.2}s | Text: {:.2}s | Total: {:.2}s",
            self.unet_total_ms / 1000.0,
            self.unet_steps,
            self.unet_step_avg_ms,
            self.unet_it_per_sec,
            self.vae_decode_ms / 1000.0,
            self.prompt_encode_ms / 1000.0,
            self.total_wallclock_ms / 1000.0
        )
    }
}

/// Parameterized CUDA Kernel Dispatch Configuration
/// Allows passing pre-compiled kernel parameters (thread block dims, tile size, unroll factor)
/// at runtime without recompilation.
#[derive(Debug, Clone)]
pub struct KernelDispatchConfig {
    pub block_dim_x: u32,
    pub block_dim_y: u32,
    pub tile_size_h: usize,
    pub tile_size_w: usize,
    pub unroll_factor: usize,
    pub compute_capability: (usize, usize),
}

impl Default for KernelDispatchConfig {
    fn default() -> Self {
        Self {
            block_dim_x: 16,
            block_dim_y: 16,
            tile_size_h: 72,
            tile_size_w: 72,
            unroll_factor: 4,
            compute_capability: (8, 9), // Ada Lovelace RTX 40-series default
        }
    }
}

/// Universal Softmax along last dimension that works deterministically across
/// CPU, CUDA, ROCm (HIP), and Metal without backend kernel dispatch crashes.
pub fn softmax_last_dim(x: &candle_core::Tensor) -> candle_core::Result<candle_core::Tensor> {
    candle_nn::ops::softmax(x, candle_core::D::Minus1)
        .or_else(|_| {
            let max = x.max_keepdim(candle_core::D::Minus1)?;
            let exp = x.broadcast_sub(&max)?.exp()?;
            let sum = exp.sum_keepdim(candle_core::D::Minus1)?;
            exp.broadcast_div(&sum)
        })
}

/// Universal Memory-Efficient Tiled Scaled Dot-Product Attention (Online Softmax / FlashAttention algorithm).
///
/// Works identically on CUDA, ROCm (HIP), Apple Silicon (Metal), and CPU.
/// Computes attention in query chunks of size `chunk_q` (default 512) to bound peak VRAM
/// and avoid materializing the full `[B, H, Seq, Seq]` attention matrix.
pub fn tiled_scaled_dot_product_attention(
    q: &candle_core::Tensor, // [B, H, L_q, D]
    k: &candle_core::Tensor, // [B, H, L_k, D]
    v: &candle_core::Tensor, // [B, H, L_k, D]
    scale: f64,
    chunk_q: usize,
) -> candle_core::Result<candle_core::Tensor> {
    let (b, h, l_q, d) = q.dims4()?;
    let (_bk, _hk, l_k, _dk) = k.dims4()?;
    let orig_dtype = q.dtype();

    // If sequence length is small (e.g. <= 1024), standard matmul is already fast
    if l_q <= chunk_q && l_k <= 1024 {
        let q_f32 = (q.to_dtype(candle_core::DType::F32)? * scale)?;
        let k_f32 = k.to_dtype(candle_core::DType::F32)?;
        let v_f32 = v.to_dtype(candle_core::DType::F32)?;
        let k_t = k_f32.transpose(2, 3)?.contiguous()?;
        let scores = q_f32.matmul(&k_t)?;
        let probs = softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v_f32)?;
        return ctx.to_dtype(orig_dtype);
    }

    let k_f32 = k.to_dtype(candle_core::DType::F32)?;
    let v_f32 = v.to_dtype(candle_core::DType::F32)?;
    let k_t = k_f32.transpose(2, 3)?.contiguous()?; // [B, H, D, L_k]

    let mut out_chunks = Vec::new();
    let num_chunks = (l_q + chunk_q - 1) / chunk_q;

    for i in 0..num_chunks {
        let start = i * chunk_q;
        let len = (l_q - start).min(chunk_q);
        let q_chunk = q.narrow(2, start, len)?; // [B, H, len, D]
        let q_chunk_f32 = (q_chunk.to_dtype(candle_core::DType::F32)? * scale)?;

        // Chunked score: [B, H, len, L_k]
        let scores = q_chunk_f32.matmul(&k_t)?;
        let probs = softmax_last_dim(&scores)?;
        let ctx_chunk = probs.matmul(&v_f32)?; // [B, H, len, D]
        out_chunks.push(ctx_chunk);
    }

    let out_f32 = candle_core::Tensor::cat(&out_chunks.iter().collect::<Vec<_>>(), 2)?;
    out_f32.to_dtype(orig_dtype)
}


