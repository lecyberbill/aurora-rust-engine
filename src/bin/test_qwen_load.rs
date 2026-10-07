use std::path::Path;
use candle_core::{DType, Device};
use aurora_rust_engine::weights::SafeTensorsArchive;
use aurora_rust_engine::text::Qwen3TextEncoder;

fn main() -> anyhow::Result<()> {
    let vae_path = Path::new("/models/comfyui/vae/qwen_image_vae_complet.safetensors");
    let vae_archive = SafeTensorsArchive::open(vae_path)?;
    println!("VAE Archive keys: {}", vae_archive.tensor_names().len());
    for k in vae_archive.tensor_names() {
        if k.contains("resample") || k.contains("conv_in") || k.contains("conv_out") || k.contains("upsamplers") {
            if let Some((dt, shape)) = vae_archive.raw_info(&k) {
                println!("🔎 {} | dtype={:?} | shape={:?}", k, dt, shape);
            }
        }
    }
    
    Ok(())
}
