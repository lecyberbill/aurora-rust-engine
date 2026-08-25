// [WFGY] Zone: SAFE | λ: 0.15 | Fallbacks: 0 | Action: Debug Step 1 of Transformer forward pass layer by layer

use candle_core::{DType, Device, Tensor};
use aurora_rust_engine::weights::SafeTensorsArchive;
use aurora_rust_engine::diffusion::dit::{FluxConfig, FluxTransformer};
use aurora_rust_engine::weights::WeightRouter;
use std::sync::Arc;

fn main() -> anyhow::Result<()> {
    println!("🔍 Debugging Step 1 Transformer Forward Pass Layer-by-Layer...");

    let device = Device::new_cuda(0)?;
    let checkpoint = "G:\\models\\flux\\flux1SchnellFp8_schnellFp8.safetensors";
    let archive = Arc::new(SafeTensorsArchive::open(checkpoint)?);
    let router = WeightRouter::new(&archive, device.clone(), DType::F16);
    let vb = router.flux_header_var_builder()?;
    let config = FluxConfig::schnell();
    let transformer = FluxTransformer::new_streaming(config.clone(), vb)?;

    let streamer = aurora_rust_engine::diffusion::dit::streamer::SequentialBlockStreamer::new(
        archive.clone(),
        device.clone(),
        DType::F16,
        config.hidden_size,
        config.num_heads,
        config.mlp_ratio,
    );

    let latents = Tensor::randn(0f32, 1f32, (1, 1024, 64), &device)?.to_dtype(DType::F16)?;
    let txt = Tensor::randn(0f32, 1f32, (1, 256, 4096), &device)?.to_dtype(DType::F16)?;
    let t_tensor = Tensor::from_slice(&[1.0f32], (1,), &device)?.to_dtype(DType::F16)?;
    let y_vec = Tensor::randn(0f32, 1f32, (1, 768), &device)?.to_dtype(DType::F16)?;

    println!("⚡ Executing Transformer forward pass with streamer...");
    let velocity = transformer.forward_with_streamer(
        &latents,
        &txt,
        &t_tensor,
        Some(&y_vec),
        None,
        Some(&streamer),
    )?;

    let v_vec = velocity.flatten_all()?.to_dtype(DType::F32)?.to_device(&Device::Cpu)?.to_vec1::<f32>()?;
    let min_v = v_vec.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_v = v_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean_v = v_vec.iter().sum::<f32>() / v_vec.len() as f32;

    println!("📊 Velocity Stats: Min = {:.4}, Max = {:.4}, Mean = {:.4}", min_v, max_v, mean_v);

    Ok(())
}
