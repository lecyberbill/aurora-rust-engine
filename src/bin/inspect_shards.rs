// Inspect a multi-shard safetensors checkpoint: aggregate dims of key tensors across shards.
use std::sync::Arc;
use aurora_rust_engine::weights::SafeTensorsArchive;

fn main() -> anyhow::Result<()> {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    if paths.is_empty() {
        eprintln!("usage: inspect_shards <shard1.safetensors> [shard2 ...]");
        std::process::exit(2);
    }
    let refs: Vec<&str> = paths.iter().map(|s| s.as_str()).collect();
    let archive = Arc::new(SafeTensorsArchive::open_shards(&refs)?);
    println!("opened {} shards, {} tensors", paths.len(), archive.keys().count());

    let target = ["model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight", "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.mlp.gate_proj.weight", "model.layers.0.mlp.down_proj.weight"];
    for key in archive.keys() {
        if target.contains(&key.as_str()) {
            if let Some((d, s)) = archive.raw_info(key) {
                println!("   {:<60} dims={:?} dtype={:?}", key, s, d);
            }
        }
    }
    // layer count
    let mut layers = 0;
    for key in archive.keys() {
        if let Some(r) = key.strip_prefix("model.layers.") {
            if let Some(i) = r.split('.').next().and_then(|s| s.parse::<usize>().ok()) {
                if i + 1 > layers { layers = i + 1; }
            }
        }
    }
    println!("model.layers count = {}", layers);
    Ok(())
}
