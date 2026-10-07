use std::path::Path;
use candle_core::{DType, Device};
use aurora_rust_engine::device::auto_device;
use aurora_rust_engine::pipelines::ZImageTurboPipeline;

fn main() -> anyhow::Result<()> {
    std::env::set_var("RUST_BACKTRACE", "1");
    let device = auto_device()?;
    let dtype = if device.is_cuda() { DType::BF16 } else { DType::F32 };
    let path_str = std::env::args().nth(1).unwrap_or_else(|| r"G:\models\zit\z-image-turbo-fp8-aio.safetensors".to_string());
    let vae_path_str = std::env::args().nth(2).unwrap_or_else(|| r"G:\models\zit\zImageTurbo_vae.safetensors".to_string());
    let path = Path::new(&path_str);
    let vae_path = Path::new(&vae_path_str);
    
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
