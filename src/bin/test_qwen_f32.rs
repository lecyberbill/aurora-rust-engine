use aurora_rust_engine::weights::SafeTensorsArchive;
use aurora_rust_engine::text::Qwen3TextEncoder;
use candle_core::{DType, Device, Module, Tensor};
use std::path::PathBuf;

fn print_stats(name: &str, t: &Tensor) -> candle_core::Result<()> {
    let t_f32 = t.to_dtype(DType::F32)?;
    let shape = t.dims();
    let mean_t = t_f32.mean_all()?;
    let mean = mean_t.to_scalar::<f32>()?;
    let diff = t_f32.broadcast_sub(&mean_t)?;
    let var = diff.sqr()?.mean_all()?.to_scalar::<f32>()?;
    let std = var.sqrt();
    let min = t_f32.flatten_all()?.to_vec1::<f32>()?.into_iter().fold(f32::INFINITY, f32::min);
    let max = t_f32.flatten_all()?.to_vec1::<f32>()?.into_iter().fold(f32::NEG_INFINITY, f32::max);
    println!("   📊 [{:<35}] shape={:?} | mean={:+.4} | std={:.4} | min={:+.4} | max={:+.4}", name, shape, mean, std, min, max);
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let te_path = PathBuf::from("/models/comfyui/text_encoders/qwen3vl_4b_fp8_scaled.safetensors");
    let archive = SafeTensorsArchive::open(&te_path)?;
    let tok_path = PathBuf::from("qwen_tokenizer_KREA2.json");
    
    let emb_w = archive.get_tensor("model.embed_tokens.weight", &device, DType::F32)?;
    print_stats("model.embed_tokens.weight", &emb_w)?;

    let encoder = Qwen3TextEncoder::from_archive(&archive, Some(&tok_path), &device, DType::F32)?;
    let prompt = "a majestic white wolf on a snowy cliff at golden hour, photorealistic, 8k";
    let toks = encoder.encode_krea2_12_layers(prompt, 0)?;
    print_stats("Final 12 taps tensor", &toks)?;
    Ok(())
}
