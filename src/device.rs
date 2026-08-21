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
