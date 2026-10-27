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

/// Pinned Host Memory buffer (Page-Locked RAM) allowing zero-copy direct DMA transfers to GPU.
pub struct PinnedHostBuffer {
    ptr: *mut u8,
    size: usize,
}

unsafe impl Send for PinnedHostBuffer {}
unsafe impl Sync for PinnedHostBuffer {}

impl PinnedHostBuffer {
    pub fn allocate(size: usize) -> Option<Self> {
        #[cfg(feature = "rocm")]
        {
            extern "C" {
                fn hip_alloc_pinned_host(size: usize) -> *mut u8;
            }
            let ptr = unsafe { hip_alloc_pinned_host(size) };
            if !ptr.is_null() {
                return Some(Self { ptr, size });
            }
        }

        // Standard CPU page-aligned fallback
        let layout = std::alloc::Layout::from_size_align(size, 4096).ok()?;
        let ptr = unsafe { std::alloc::alloc(layout) };
        if !ptr.is_null() {
            Some(Self { ptr, size })
        } else {
            None
        }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.size) }
    }

    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.size) }
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    pub fn size(&self) -> usize {
        self.size
    }
}

impl Drop for PinnedHostBuffer {
    fn drop(&mut self) {
        #[cfg(feature = "rocm")]
        {
            extern "C" {
                fn hip_free_pinned_host(ptr: *mut u8);
            }
            unsafe { hip_free_pinned_host(self.ptr) };
            return;
        }

        #[allow(unreachable_code)]
        {
            if let Ok(layout) = std::alloc::Layout::from_size_align(self.size, 4096) {
                unsafe { std::alloc::dealloc(self.ptr, layout) };
            }
        }
    }
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
    // Fused single-kernel softmax when the backend has it (ROCm/CUDA/Metal).
    // The generic fallback uses reduce ops, which copy metadata to the device on
    // every call — not capturable in a HIP graph and slower in decode.
    candle_nn::ops::softmax_last_dim(x).or_else(|_| {
        candle_nn::ops::softmax(x, candle_core::D::Minus1)
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
    #[allow(unused_variables)]
    let (b, h, l_q, d) = q.dims4()?;
    let (_bk, _hk, l_k, _dk) = k.dims4()?;
    let orig_dtype = q.dtype();

    #[cfg(feature = "rocm")]
    {
        extern "C" {
            fn launch_hip_fused_sdpa_f16(
                q: *const std::ffi::c_void,
                k: *const std::ffi::c_void,
                v: *const std::ffi::c_void,
                out: *mut std::ffi::c_void,
                batch_size: i32,
                num_heads: i32,
                seq_q: i32,
                seq_k: i32,
                head_dim: i32,
                scale: f32,
                stream: *mut std::ffi::c_void,
            );
        }

        // On ROCm device with F16, dispatch to native fused kernel
        if q.device().is_rocm() && orig_dtype == candle_core::DType::F16 && d <= 128 {
            let q_c = q.contiguous()?;
            let k_c = k.contiguous()?;
            let v_c = v.contiguous()?;
            let out = candle_core::Tensor::zeros((b, h, l_q, d), orig_dtype, q.device())?;

            let success = {
                let (q_st, _) = q_c.storage_and_layout();
                let (k_st, _) = k_c.storage_and_layout();
                let (v_st, _) = v_c.storage_and_layout();
                let (out_st, _) = out.storage_and_layout();

                if let (
                    candle_core::Storage::Rocm(q_rocm),
                    candle_core::Storage::Rocm(k_rocm),
                    candle_core::Storage::Rocm(v_rocm),
                    candle_core::Storage::Rocm(out_rocm),
                ) = (&*q_st, &*k_st, &*v_st, &*out_st) {
                    let q_ptr = q_rocm.slice.as_ptr();
                    let k_ptr = k_rocm.slice.as_ptr();
                    let v_ptr = v_rocm.slice.as_ptr();
                    let out_ptr = out_rocm.slice.as_ptr() as *mut std::ffi::c_void;

                    let stream = match q.device() {
                        candle_core::Device::Rocm(dev) => {
                            dev.stream().as_raw() as *mut std::ffi::c_void
                        }
                        _ => std::ptr::null_mut(),
                    };
                    unsafe {
                        launch_hip_fused_sdpa_f16(
                            q_ptr,
                            k_ptr,
                            v_ptr,
                            out_ptr,
                            b as i32,
                            h as i32,
                            l_q as i32,
                            l_k as i32,
                            d as i32,
                            scale as f32,
                            stream,
                        );
                    }
                    true
                } else {
                    false
                }
            };

            if success {
                return Ok(out);
            }
        }
    }

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

/// Split-K flash-decoding attention for a single query per head (ROCm, F32, D=256).
///
/// `k`/`v` are the *unique* GQA tensors `[B, Hkv, L, D]`; head `h` reads kv head
/// `h / (H / Hkv)`, so no repeat/transpose copy is needed. Returns `None` when
/// the shape or backend is unsupported so the caller keeps the portable path.
#[cfg(feature = "rocm")]
fn flash_decode_impl(
    q: &candle_core::Tensor,
    k: &candle_core::Tensor,
    v: &candle_core::Tensor,
    scale: f64,
    mask: Option<&candle_core::Tensor>,
) -> candle_core::Result<Option<candle_core::Tensor>> {
    use candle_core::rocm_backend::RocmStorageSlice;
    use candle_core::{DType, Device, Storage};

    extern "C" {
        #[allow(clippy::too_many_arguments)]
        fn launch_flash_decode_f32(
            q: *const f32,
            k: *const f32,
            v: *const f32,
            mask: *const f32,
            pm: *mut f32,
            pl: *mut f32,
            pacc: *mut f32,
            out: *mut f32,
            b: i32,
            h: i32,
            hkv: i32,
            l: i32,
            nc: i32,
            ck: i32,
            scale: f32,
            stream: *mut std::ffi::c_void,
        );
    }

    if std::env::var_os("AURORA_NO_FLASH").is_some() {
        return Ok(None);
    }
    if !q.device().is_rocm() || q.dtype() != DType::F32 {
        return Ok(None);
    }
    let (b, h, lq, d) = q.dims4()?;
    let (_bk, hkv, l, _dk) = k.dims4()?;
    if lq != 1 || d != 256 || h == 0 || hkv == 0 || h % hkv != 0 {
        return Ok(None);
    }

    fn ptr_of(t: &candle_core::Tensor) -> candle_core::Result<*mut std::ffi::c_void> {
        let (st, layout) = t.storage_and_layout();
        let off = layout.start_offset();
        match &*st {
            Storage::Rocm(r) => match &r.slice {
                RocmStorageSlice::F32(m) => Ok(unsafe { m.ptr_at(off) }),
                _ => candle_core::bail!("flash: not f32"),
            },
            _ => candle_core::bail!("flash: not rocm"),
        }
    }

    let ck = 64usize;
    let nc = l.div_ceil(ck);
    let dev = q.device().clone();
    // The KV cache can hand back a non-contiguous `v` (it is a transposed view
    // when the prompt prefill stored it). The flat kernel needs packed inputs.
    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let mask_t = match mask {
        Some(m) => {
            if m.dtype() != DType::F32 || m.elem_count() != l {
                return Ok(None);
            }
            Some(m.contiguous()?)
        }
        None => None,
    };
    let mask_ptr = match &mask_t {
        Some(m) => ptr_of(m)? as *const f32,
        None => std::ptr::null(),
    };
    let pm = candle_core::Tensor::zeros((b * h * nc,), DType::F32, &dev)?;
    let pl = candle_core::Tensor::zeros((b * h * nc,), DType::F32, &dev)?;
    let pacc = candle_core::Tensor::zeros((b * h * nc, d), DType::F32, &dev)?;
    let out = candle_core::Tensor::zeros((b * h, d), DType::F32, &dev)?;

    let stream = match &dev {
        Device::Rocm(d) => d.stream().as_raw() as *mut std::ffi::c_void,
        _ => return Ok(None),
    };

    unsafe {
        launch_flash_decode_f32(
            ptr_of(&q)? as *const f32,
            ptr_of(&k)? as *const f32,
            ptr_of(&v)? as *const f32,
            mask_ptr,
            ptr_of(&pm)? as *mut f32,
            ptr_of(&pl)? as *mut f32,
            ptr_of(&pacc)? as *mut f32,
            ptr_of(&out)? as *mut f32,
            b as i32,
            h as i32,
            hkv as i32,
            l as i32,
            nc as i32,
            ck as i32,
            scale as f32,
            stream,
        );
    }
    Ok(Some(out.reshape((b, h, 1, d))?))
}

#[cfg(feature = "rocm")]
pub fn flash_decode(
    q: &candle_core::Tensor,
    k: &candle_core::Tensor,
    v: &candle_core::Tensor,
    scale: f64,
) -> candle_core::Result<Option<candle_core::Tensor>> {
    flash_decode_impl(q, k, v, scale, None)
}

/// Like [`flash_decode`] but adds a per-KV-position additive bias `mask` (`[L]`,
/// F32). Used by the static HIP graph, which attends over the full `max_len`.
#[cfg(feature = "rocm")]
pub fn flash_decode_masked(
    q: &candle_core::Tensor,
    k: &candle_core::Tensor,
    v: &candle_core::Tensor,
    scale: f64,
    mask: &candle_core::Tensor,
) -> candle_core::Result<Option<candle_core::Tensor>> {
    flash_decode_impl(q, k, v, scale, Some(mask))
}

#[cfg(not(feature = "rocm"))]
pub fn flash_decode(
    _q: &candle_core::Tensor,
    _k: &candle_core::Tensor,
    _v: &candle_core::Tensor,
    _scale: f64,
) -> candle_core::Result<Option<candle_core::Tensor>> {
    Ok(None)
}

#[cfg(not(feature = "rocm"))]
pub fn flash_decode_masked(
    _q: &candle_core::Tensor,
    _k: &candle_core::Tensor,
    _v: &candle_core::Tensor,
    _scale: f64,
    _mask: &candle_core::Tensor,
) -> candle_core::Result<Option<candle_core::Tensor>> {
    Ok(None)
}



#[cfg(feature = "rocm")]
pub fn gdn_step(
    state: &candle_core::Tensor,
    q: &candle_core::Tensor,
    k: &candle_core::Tensor,
    v: &candle_core::Tensor,
    decay: &candle_core::Tensor,
    beta: &candle_core::Tensor,
) -> candle_core::Result<Option<candle_core::Tensor>> {
    use candle_core::rocm_backend::RocmStorageSlice;
    use candle_core::{DType, Device, Storage};

    extern "C" {
        #[allow(clippy::too_many_arguments)]
        fn launch_gdn_step_f32(
            state: *mut f32,
            q: *const f32,
            k: *const f32,
            v: *const f32,
            decay: *const f32,
            beta: *const f32,
            out: *mut f32,
            h: i32,
            dv: i32,
            dk: i32,
            stream: *mut std::ffi::c_void,
        );
    }

    if std::env::var_os("AURORA_NO_GDN").is_some() {
        return Ok(None);
    }
    if !state.device().is_rocm() || state.dtype() != DType::F32 {
        return Ok(None);
    }
    let (h, dv, dk) = state.dims3()?;
    if dk != 128 {
        return Ok(None);
    }
    let (_, sl) = state.storage_and_layout();
    if !sl.is_contiguous() {
        return Ok(None);
    }
    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let decay = decay.contiguous()?;
    let beta = beta.contiguous()?;
    let out = candle_core::Tensor::zeros((h, dv), DType::F32, state.device())?;

    fn ptr_of(t: &candle_core::Tensor) -> candle_core::Result<*mut std::ffi::c_void> {
        let (st, layout) = t.storage_and_layout();
        let off = layout.start_offset();
        match &*st {
            Storage::Rocm(r) => match &r.slice {
                RocmStorageSlice::F32(m) => Ok(unsafe { m.ptr_at(off) }),
                _ => candle_core::bail!("gdn: not f32"),
            },
            _ => candle_core::bail!("gdn: not rocm"),
        }
    }
    let stream = match state.device() {
        Device::Rocm(d) => d.stream().as_raw() as *mut std::ffi::c_void,
        _ => return Ok(None),
    };
    unsafe {
        launch_gdn_step_f32(
            ptr_of(state)? as *mut f32,
            ptr_of(&q)? as *const f32,
            ptr_of(&k)? as *const f32,
            ptr_of(&v)? as *const f32,
            ptr_of(&decay)? as *const f32,
            ptr_of(&beta)? as *const f32,
            ptr_of(&out)? as *mut f32,
            h as i32,
            dv as i32,
            dk as i32,
            stream,
        );
    }
    Ok(Some(out))
}

#[cfg(not(feature = "rocm"))]
pub fn gdn_step(
    _state: &candle_core::Tensor,
    _q: &candle_core::Tensor,
    _k: &candle_core::Tensor,
    _v: &candle_core::Tensor,
    _decay: &candle_core::Tensor,
    _beta: &candle_core::Tensor,
) -> candle_core::Result<Option<candle_core::Tensor>> {
    Ok(None)
}
