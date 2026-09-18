use std::sync::Arc;
use aurora_rust_engine::weights::SafeTensorsArchive;

fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).expect("usage: dump_prefixes <path.safetensors>");
    let archive = Arc::new(SafeTensorsArchive::open(&path)?);
    let cpu = candle_core::Device::Cpu;

    let mut keys: Vec<String> = archive.keys().cloned().collect();
    keys.sort();

    println!("=== TOTAL TENSORS: {} ===", keys.len());

    // Print categories
    let mut prefixes = std::collections::BTreeSet::new();
    for k in &keys {
        let parts: Vec<&str> = k.split('.').collect();
        if parts.len() >= 3 {
            prefixes.insert(format!("{}.{}.{}", parts[0], parts[1], parts[2]));
        }
    }
    println!("=== PREFIXES ===");
    for p in prefixes {
        println!("  {}", p);
    }

    println!("\n=== DIT LAYER 0 & EMBEDDERS ===");
    for k in &keys {
        if k.starts_with("model.diffusion_model.layers.0.")
            || k.starts_with("model.diffusion_model.t_embedder.")
            || k.starts_with("model.diffusion_model.x_embedder.")
            || k.starts_with("model.diffusion_model.cap_")
            || k.starts_with("model.diffusion_model.context_")
            || k.starts_with("model.diffusion_model.final_layer.")
            || k.starts_with("model.diffusion_model.norm_final")
            || k.starts_with("text_encoders.qwen3_4b.transformer.model.embed_tokens")
        {
            if let Ok(t) = archive.get_tensor(k, &cpu, candle_core::DType::F32) {
                println!("  {:<65} dims={:?} dtype={:?}", k, t.dims(), t.dtype());
            }
        }
    }

    Ok(())
}
