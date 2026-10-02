use std::path::Path;
use candle_core::{DType, Device};
use aurora_rust_engine::device::auto_device;
use aurora_rust_engine::pipelines::ZImageTurboPipeline;

fn main() -> anyhow::Result<()> {
    std::env::set_var("RUST_BACKTRACE", "1");
    let device = auto_device()?;
    let dtype = if device.is_cuda() { DType::BF16 } else { DType::F32 };
    let path = Path::new("/models/comfyui/diffusion_models/krea2_turbo_fp8_scaled.safetensors");
    let vae_path = Path::new("/models/comfyui/vae/qwen_image_vae_complet.safetensors");
    
    println!("Loading DiT pipeline from {:?}...", path);
    let pipeline = ZImageTurboPipeline::from_aio_checkpoint(
        path,
        Some(vae_path),
        &device,
        dtype,
    );
    match pipeline {
        Ok(_) => println!("✅ Pipeline loaded successfully!"),
        Err(e) => eprintln!("❌ Pipeline load error: {:?}", e),
    }
    Ok(())
}
