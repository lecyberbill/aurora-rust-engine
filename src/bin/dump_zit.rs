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

    println!("\n=== QWEN TEXT ENCODER TENSORS ===");
    for k in &keys {
        if k.contains("text_encoders.qwen3_4b") {
            if let Ok(t) = archive.get_tensor(k, &cpu, candle_core::DType::F32) {
                if k.contains("layers.0.") || k.contains("embed_tokens") || k.contains("norm") {
                    println!("  {:<65} dims={:?}", k, t.dims());
                }
            }
        }
    }

    Ok(())
}
