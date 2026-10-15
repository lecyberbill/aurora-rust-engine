use std::path::PathBuf;
use aurora_rust_engine::weights::open_auto;

fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).unwrap_or_else(|| "/models/llm/LFM2.5-8B-A1B-Q4_K_M.gguf".to_string());
    println!("Inspecting: {}", path);
    let archive = open_auto(&PathBuf::from(&path))?;
    let keys = archive.keys();
    println!("Total keys: {}", keys.len());
    
    println!("\n=== Layer 0 tensors ===");
    for k in keys.iter().filter(|k| k.contains("blk.0.")) {
        if let Some(info) = archive.raw_info(k) {
            println!("  {} -> shape: {:?}, dtype: {:?}", k, info.1, info.0);
        }
    }
    
    println!("\n=== Layer 1 tensors ===");
    for k in keys.iter().filter(|k| k.contains("blk.1.")) {
        if let Some(info) = archive.raw_info(k) {
            println!("  {} -> shape: {:?}, dtype: {:?}", k, info.1, info.0);
        }
    }

    println!("\n=== Layer 2 tensors ===");
    for k in keys.iter().filter(|k| k.contains("blk.2.")) {
        if let Some(info) = archive.raw_info(k) {
            println!("  {} -> shape: {:?}, dtype: {:?}", k, info.1, info.0);
        }
    }

    println!("\n=== Layer Types across 24 layers ===");
    for l in 0..24 {
        let has_shortconv = keys.iter().any(|k| k == &format!("blk.{l}.shortconv.conv.weight"));
        let has_attn = keys.iter().any(|k| k == &format!("blk.{l}.attn_q.weight"));
        let has_moe = keys.iter().any(|k| k == &format!("blk.{l}.ffn_gate_exps.weight"));
        println!("  Layer {:2}: conv={}, attn={}, moe={}", l, has_shortconv, has_attn, has_moe);
    }

    println!("\n=== Global / Normalization tensors ===");
    for k in keys.iter() {
        if !k.starts_with("blk.") || k.contains("output") || k.contains("norm") && !k.contains(".attn_") && !k.contains(".ffn_") {
            if let Some(info) = archive.raw_info(k) {
                println!("  {} -> shape: {:?}, dtype: {:?}", k, info.1, info.0);
            }
        }
    }

    Ok(())
}
