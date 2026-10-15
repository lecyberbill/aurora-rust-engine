use std::path::PathBuf;
use aurora_rust_engine::weights::{open_auto, WeightsSource};

fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).unwrap_or_else(|| "/models/llm/LFM2.5-8B-A1B-Q4_K_M.gguf".to_string());
    println!("Inspecting: {}", path);
    let archive = open_auto(&PathBuf::from(&path))?;
    let keys = archive.keys();
    println!("Total keys: {}", keys.len());
    
    println!("\n=== Layer 0 tensors ===");
    for k in keys.iter().filter(|k| k.contains("blk.0.")) {
        if let Ok(info) = archive.raw_info(k) {
            println!("  {} -> shape: {:?}, dtype: {:?}", k, info.shape, info.dtype);
        }
    }
    
    println!("\n=== Layer 1 tensors ===");
    for k in keys.iter().filter(|k| k.contains("blk.1.")) {
        if let Ok(info) = archive.raw_info(k) {
            println!("  {} -> shape: {:?}, dtype: {:?}", k, info.shape, info.dtype);
        }
    }

    println!("\n=== Layer 2 tensors ===");
    for k in keys.iter().filter(|k| k.contains("blk.2.")) {
        if let Ok(info) = archive.raw_info(k) {
            println!("  {} -> shape: {:?}, dtype: {:?}", k, info.shape, info.dtype);
        }
    }

    println!("\n=== Global / Normalization tensors ===");
    for k in keys.iter().filter(|k| !k.starts_with("blk.")) {
        if let Ok(info) = archive.raw_info(k) {
            println!("  {} -> shape: {:?}, dtype: {:?}", k, info.shape, info.dtype);
        }
    }

    Ok(())
}
