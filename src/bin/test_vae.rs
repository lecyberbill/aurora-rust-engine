// [WFGY] Zone: SAFE | λ: 0.10 | Fallbacks: 0 | Action: Diagnose VAE keys and structure

use candle_core::{DType, Device, Tensor};
use aurora_rust_engine::weights::{SafeTensorsArchive, WeightRouter};
use aurora_rust_engine::diffusion::vae_flux::FluxVaeDecoder;
use aurora_rust_engine::diffusion::vae::VaeDecoder;

fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).unwrap_or_else(|| "/models/comfyui/vae/qwen_image_vae.safetensors".into());
    println!("🔍 Diagnosing VAE: {:?}", path);
    let archive = SafeTensorsArchive::open(&path)?;
    let keys: Vec<String> = archive.keys().map(|k| k.to_string()).collect();
    println!("🔑 Total keys: {}", keys.len());
    let mut decoder_keys: Vec<_> = keys.iter().filter(|k| k.starts_with("decoder.")).collect();
    decoder_keys.sort();
    for k in decoder_keys {
        if let Ok(t) = archive.get_tensor(k, &Device::Cpu, DType::F32) {
            println!("   • {:<50} shape={:?}", k, t.dims());
        }
    }

    let router = WeightRouter::new(&archive, Device::Cpu, DType::F32);
    println!("\nTesting WeightRouter::vae_var_builder...");
    match router.vae_var_builder() {
        Ok(vb) => {
            println!("✅ vae_var_builder succeeded!");
            match FluxVaeDecoder::new(vb.clone()) {
                Ok(_) => println!("✅ FluxVaeDecoder successfully created!"),
                Err(e) => println!("❌ FluxVaeDecoder creation error: {:?}", e),
            }
            match VaeDecoder::new(vb, true) {
                Ok(_) => println!("✅ VaeDecoder (SDXL/SD15) successfully created!"),
                Err(e) => println!("❌ VaeDecoder creation error: {:?}", e),
            }
        }
        Err(e) => println!("❌ vae_var_builder error: {:?}", e),
    }

    Ok(())
}
