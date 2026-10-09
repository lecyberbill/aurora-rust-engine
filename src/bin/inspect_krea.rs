use aurora_rust_engine::weights::SafeTensorsArchive;
use candle_core::{DType, Device};

fn main() -> anyhow::Result<()> {
    for (name, path) in [
        ("DiT", "/models/comfyui/diffusion_models/krea2_turbo_fp8_scaled.safetensors"),
        ("Text Encoder", "/models/comfyui/text_encoders/qwen3vl_4b_fp8_scaled.safetensors"),
        ("VAE", "/models/comfyui/vae/qwen_image_vae.safetensors"),
    ] {
        println!("==================================================");
        println!("🔍 Inspecting {}: {}", name, path);
        match SafeTensorsArchive::open(path) {
            Ok(archive) => {
                let keys: Vec<String> = archive.keys().map(|k| k.to_string()).collect();
                println!("   Total keys: {}", keys.len());
                for k in &keys {
                    if !k.starts_with("blocks.") && !k.ends_with(".weight_scale") {
                        if let Ok(t) = archive.get_tensor(k, &Device::Cpu, DType::F32) {
                            println!("   • {:<50} {:?}", k, t.dims());
                        } else {
                            println!("   • {}", k);
                        }
                    }
                }
            }
            Err(e) => {
                println!("   ❌ Failed to open: {:?}", e);
            }
        }
    }
    Ok(())
}
