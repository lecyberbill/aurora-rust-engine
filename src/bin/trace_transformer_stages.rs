// [WFGY] Zone: SAFE | λ: 0.15 | Fallbacks: 0 | Action: Instrument full transformer step-by-step to find NaN stage

use candle_core::{DType, Device, Module, Tensor};
use aurora_rust_engine::weights::SafeTensorsArchive;
use aurora_rust_engine::diffusion::dit::{FluxConfig, FluxTransformer};
use aurora_rust_engine::weights::WeightRouter;
use std::sync::Arc;

fn print_tensor(name: &str, t: &Tensor) -> anyhow::Result<()> {
    let v = t.flatten_all()?.to_dtype(DType::F32)?.to_device(&Device::Cpu)?.to_vec1::<f32>()?;
    let min = v.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let has_nan = v.iter().any(|x| x.is_nan());
    println!("  {}: Min = {:.4}, Max = {:.4}, Has NaN = {}", name, min, max, has_nan);
    Ok(())
}

fn main() -> anyhow::Result<()> {
    println!("🔍 Step-by-Step Transformer Stage Probe...");

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

    let latents = (Tensor::randn(0f32, 1f32, (1, 1024, 64), &device)? * 0.1)?.to_dtype(DType::F16)?;
    let txt = (Tensor::randn(0f32, 1f32, (1, 256, 4096), &device)? * 0.1)?.to_dtype(DType::F16)?;
    let t_tensor = Tensor::from_slice(&[1.0f32], (1,), &device)?.to_dtype(DType::F16)?;
    let y_vec = (Tensor::randn(0f32, 1f32, (1, 768), &device)? * 0.1)?.to_dtype(DType::F16)?;

    println!("⚡ 1. Checking temb & Projections...");
    let mut temb = transformer.time_embedder.forward(&t_tensor)?;
    print_tensor("temb (after time_in)", &temb)?;

    let mut img_h = transformer.img_in.forward(&latents)?;
    let mut txt_h = transformer.txt_in.forward(&txt)?;
    print_tensor("img_h (after img_in)", &img_h)?;
    print_tensor("txt_h (after txt_in)", &txt_h)?;

    println!("⚡ 2. Checking Double Stream Blocks...");
    for i in 0..config.num_double_blocks {
        let (next_img, next_txt) = streamer.execute_double_block(i, &img_h, &txt_h, &temb, None, None)?;
        img_h = next_img;
        txt_h = next_txt;
        if i == 0 || i == 18 {
            print_tensor(&format!("double_block_{}_img", i), &img_h)?;
        }
    }

    println!("⚡ 3. Checking Single Stream Blocks...");
    let mut unified = Tensor::cat(&[&txt_h, &img_h], 1)?;
    for i in 0..config.num_single_blocks {
        unified = streamer.execute_single_block(i, &unified, &temb)?;
        if i == 0 || i == 37 {
            print_tensor(&format!("single_block_{}_unified", i), &unified)?;
        }
    }

    let txt_len = txt_h.dim(1)?;
    img_h = unified.narrow(1, txt_len, img_h.dim(1)?)?;
    print_tensor("img_h (after single_blocks narrow)", &img_h)?;

    let velocity = transformer.forward_with_streamer(
        &latents,
        &txt,
        &t_tensor,
        Some(&y_vec),
        None,
        Some(&streamer),
    )?;
    print_tensor("Final Velocity", &velocity)?;

    Ok(())
}
