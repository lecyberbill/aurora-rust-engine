// [WFGY] Zone: SAFE | λ: 0.15 | Fallbacks: 1 (CUDA/Metal fallback to CPU) | Action: Device selection and compute capability probing

use candle_core::Device;
use tracing::info;

/// Probe and select the most performant available compute device.
/// Priority: CUDA -> Metal -> CPU
pub fn auto_device() -> candle_core::Result<Device> {
    #[cfg(feature = "cuda")]
    {
        match Device::new_cuda(0) {
            Ok(device) => {
                info!("Using CUDA acceleration device (ordinal 0)");
                return Ok(device);
            }
            Err(err) => {
                tracing::warn!("CUDA device requested but unavailable: {:?}. Falling back.", err);
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
