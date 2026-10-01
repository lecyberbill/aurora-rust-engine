use aurora_rust_engine::weights::SafeTensorsArchive;
use candle_core::{DType, Device};

fn main() -> anyhow::Result<()> {
    let p = "/models/comfyui/vae/qwen_image_vae.safetensors";
    let archive = SafeTensorsArchive::open(p)?;
    let mut keys: Vec<String> = archive.keys().map(|k| k.to_string()).collect();
    keys.sort();
    println!("=== All Decoder Keys in qwen_image_vae.safetensors ===");
    for k in &keys {
        if k.starts_with("decoder.") {
            if let Ok(t) = archive.get_tensor(k, &Device::Cpu, DType::F32) {
                println!("  {:<45} {:?}", k, t.dims());
            }
        }
    }
    Ok(())
}
