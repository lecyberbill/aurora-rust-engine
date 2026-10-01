use aurora_rust_engine::weights::SafeTensorsArchive;
use candle_core::{DType, Device};

fn main() -> anyhow::Result<()> {
    let p = std::env::args().nth(1)
        .unwrap_or_else(|| "/models/comfyui/vae/qwen_image_vae.safetensors".to_string());
    let archive = SafeTensorsArchive::open(&p)?;
    let mut keys: Vec<String> = archive.keys().map(|k| k.to_string()).collect();
    keys.sort();
    println!("=== All Keys in {} (total {}) ===", p, keys.len());
    for k in keys.iter().take(30) {
        if let Ok(t) = archive.get_tensor(k, &Device::Cpu, DType::F32) {
            println!("  {:<50} {:?}", k, t.dims());
        }
    }
    for k in &keys {
        if k.starts_with("decoder.") {
            if let Ok(t) = archive.get_tensor(k, &Device::Cpu, DType::F32) {
                println!("  {:<45} {:?}", k, t.dims());
            }
        }
    }
    Ok(())
}
