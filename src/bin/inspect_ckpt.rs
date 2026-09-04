use std::sync::Arc;
use aurora_rust_engine::weights::SafeTensorsArchive;

fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).expect("usage: inspect_ckpt <path.safetensors>");
    let archive = Arc::new(SafeTensorsArchive::open(&path)?);

    let mut double = 0; let mut single = 0; let mut double_prefix = 0; let mut single_prefix = 0;
    let _f8 = 0; let _f16 = 0; let _bf16 = 0; let _f32 = 0;
    let mut keys_all = Vec::new();

    for key in archive.keys() {
        keys_all.push(key.clone());
        let k = key.as_str();
        if let Some(r) = k.strip_prefix("double_blocks.") {
            double = double.max(r.split('.').next().and_then(|s| s.parse::<usize>().ok()).map(|i| i+1).unwrap_or(0));
        }
        if let Some(r) = k.strip_prefix("model.diffusion_model.double_blocks.") { double_prefix = double_prefix.max(r.split('.').next().and_then(|s| s.parse::<usize>().ok()).map(|i| i+1).unwrap_or(0)); }
        if let Some(r) = k.strip_prefix("single_blocks.") { single = single.max(r.split('.').next().and_then(|s| s.parse::<usize>().ok()).map(|i| i+1).unwrap_or(0)); }
        if let Some(r) = k.strip_prefix("model.diffusion_model.single_blocks.") { single_prefix = single_prefix.max(r.split('.').next().and_then(|s| s.parse::<usize>().ok()).map(|i| i+1).unwrap_or(0)); }
        if k.contains("guidance_in") { println!("HAS guidance_in"); }
    }

    // dtype + shape mapping via get_tensor header info captured in keys_all
    println!("double_blocks (no prefix) max = {}", double);
    println!("model.diffusion_model.double_blocks max = {}", double_prefix);
    println!("single_blocks (no prefix) max = {}", single);
    println!("model.diffusion_model.single_blocks max = {}", single_prefix);
    println!("total keys = {}", keys_all.len());

    // Print dims (loaded to CPU) of header/global tensors
    let cpu = candle_core::Device::Cpu;
    for key in &keys_all {
        let k: &str = key.as_str();
        if k.contains("txt_in") || k.contains("img_in") || k.contains("final_layer") || k.contains("modulation") || k.contains("time_in") || k.contains("vector_in") || k.contains("guidance_in") {
            match archive.get_tensor(k, &cpu, candle_core::DType::F32) {
                Ok(t) => println!("   {:<70} dims={:?} dtype={:?}", k, t.dims(), t.dtype()),
                Err(e) => println!("   {:<70} ERROR {}", k, e),
            }
        }
    }
    // Also sample a double & single block weight to confirm hidden_dim
    for key in &keys_all {
        let k: &str = key.as_str();
        if k.starts_with("double_blocks.0") && k.contains("qkv.weight") {
            if let Ok(t) = archive.get_tensor(k, &cpu, candle_core::DType::F32) { println!("   [db0 qkv ] {:<60} dims={:?}", k, t.dims()); }
            if let Some((rd, rs)) = archive.raw_info(k) { println!("        raw_dtype={:?} raw_shape={:?}", rd, rs); }
        }
        if k.starts_with("double_blocks.0") && k.contains("qkv.weight_scale") {
            if let Ok(t) = archive.get_tensor(k, &cpu, candle_core::DType::F32) { println!("        qkv.weight_scale value={:?}", t.to_vec1::<f32>().unwrap_or_default()); }
        }
        if k.starts_with("double_blocks.0") && k.contains("txt_attn.qkv.weight") {
            if let Ok(t) = archive.get_tensor(k, &cpu, candle_core::DType::F32) { println!("   [db0 txtqkv] {:<60} dims={:?}", k, t.dims()); }
        }
        if k == "model.embed_tokens.weight" || k == "embed_tokens.weight" {
            if let Ok(t) = archive.get_tensor(k, &cpu, candle_core::DType::F32) { println!("   [emb] {:<60} dims={:?}", k, t.dims()); }
            if let Some((rd, rs)) = archive.raw_info(k) { println!("        raw_dtype={:?} raw_shape={:?}", rd, rs); }
        }
        if k == "model.layers.0.self_attn.q_proj.weight" || k == "model.layers.0.self_attn.o_proj.weight" || k == "model.layers.0.mlp.gate_proj.weight" || k == "model.layers.0.mlp.down_proj.weight" {
            if let Ok(t) = archive.get_tensor(k, &cpu, candle_core::DType::F32) { println!("   [m0] {:<60} dims={:?}", k, t.dims()); }
            let raw11 = k.strip_suffix(".weight").unwrap_or(k);
            let sk = format!("{}.weight_scale", raw11);
            println!("        raw_dtype={:?} weight_scale_present={} input_scale_present={}",
                archive.raw_info(k).map(|(d,_)| d),
                archive.raw_info(&sk).is_some(),
                archive.raw_info(&format!("{}.input_scale", raw11)).is_some());
        }
        if k.starts_with("double_blocks.0.img_attn.norm.") || k.starts_with("single_blocks.0.norm.") {
            if let Ok(t) = archive.get_tensor(k, &cpu, candle_core::DType::F32) { println!("   [norm] {:<60} dims={:?}", k, t.dims()); }
        }
    }
    Ok(())
}
