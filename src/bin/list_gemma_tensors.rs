use std::fs::File;
use std::io::BufReader;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let file = File::open(&args[1])?;
    let mut reader = BufReader::new(file);
    let content = candle_core::quantized::gguf_file::Content::read(&mut reader)?;
    let mut keys: Vec<_> = content.tensor_infos.keys().cloned().collect();
    keys.sort();
    let mut v_layers = Vec::new();
    let mut k_layers = Vec::new();
    for k in keys {
        if k.ends_with(".attn_v.weight") {
            v_layers.push(k.clone());
        }
        if k.ends_with(".attn_k.weight") {
            k_layers.push(k.clone());
        }
    }
    println!("Total attn_k tensors: {}", k_layers.len());
    println!("Total attn_v tensors: {}", v_layers.len());
    println!("Sample attn_k: {:?}", &k_layers[..5.min(k_layers.len())]);
    println!("Sample attn_v: {:?}", &v_layers[..5.min(v_layers.len())]);
    println!("Layers missing attn_v: ");
    for k in &k_layers {
        let v = k.replace(".attn_k.weight", ".attn_v.weight");
        if !v_layers.contains(&v) {
            println!("  missing: {v}");
        }
    }
    Ok(())
}
