// [WFGY] Zone: SAFE | λ: 0.15 | Fallbacks: 0 | Action: Debug Step 1 of Real Prompt Flux Pipeline

use candle_core::{DType, Device, Tensor};
use aurora_rust_engine::traits::DiffusionParams;
use aurora_rust_engine::pipelines::FluxPipeline;

fn print_stats(name: &str, t: &Tensor) -> anyhow::Result<()> {
    let v = t.flatten_all()?.to_dtype(DType::F32)?.to_device(&Device::Cpu)?.to_vec1::<f32>()?;
    let min = v.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean = v.iter().sum::<f32>() / v.len() as f32;
    let has_nan = v.iter().any(|x| x.is_nan());
    println!("  {}: Min = {:.4}, Max = {:.4}, Mean = {:.4}, Has NaN = {}", name, min, max, mean, has_nan);
    Ok(())
}

fn main() -> anyhow::Result<()> {
    println!("🔍 Inspecting Step 1 Real T5 & CLIP Embeddings...");

    let device = Device::new_cuda(0)?;
    let checkpoint = "G:\\models\\flux\\flux1SchnellFp8_schnellFp8.safetensors";
    let mut pipeline = FluxPipeline::from_single_file_streaming(checkpoint, device.clone())?;

    let prompt = "a magnificent cyberpunk cyber-cat with glowing blue neon visor";
    let t5_emb = pipeline.t5xxl.as_mut().unwrap().encode(prompt, 256)?;
    print_stats("Real T5-XXL Embedding", &t5_emb)?;

    let clip_vec = pipeline.clip_l.as_mut().unwrap().encode_pooled(prompt)?;
    print_stats("Real CLIP-L Pooled Vector", &clip_vec)?;

    let t5_tokens = t5_emb.to_device(&device)?.to_dtype(DType::F16)?;
    let y_vec = clip_vec.to_device(&device)?.to_dtype(DType::F16)?;

    let latents = (Tensor::randn(0f32, 1f32, (1, 1024, 64), &device)? * 0.1)?.to_dtype(DType::F16)?;
    let t_tensor = Tensor::from_slice(&[1.0f32], (1,), &device)?.to_dtype(DType::F16)?;

    println!("⚡ Executing Transformer Forward Step 1...");
    let velocity = pipeline.transformer.forward_with_streamer(
        &latents,
        &t5_tokens,
        &t_tensor,
        Some(&y_vec),
        None,
        pipeline.streamer.as_ref(),
    )?;
    print_stats("Step 1 Predicted Velocity", &velocity)?;

    Ok(())
}
