use aurora_rust_engine::weights::SafeTensorsArchive;
use candle_core::{DType, Device};

fn main() -> anyhow::Result<()> {
    let p = "/models/comfyui/diffusion_models/krea2_turbo_fp8_scaled.safetensors";
    let archive = SafeTensorsArchive::open(p)?;
    let keys: Vec<String> = archive.keys().map(|k| k.to_string()).collect();
    println!("Total keys in DiT: {}", keys.len());
    for k in &keys {
        if k.contains("txt") || k.contains("refiner") || k.contains("first") || k.contains("last") || k.contains("blocks.0.") || k.contains("blocks.27.") {
            if let Ok(t) = archive.get_tensor(k, &Device::Cpu, DType::F32) {
                println!("  {:<50} {:?}", k, t.dims());
            }
        }
    }
    Ok(())
}
