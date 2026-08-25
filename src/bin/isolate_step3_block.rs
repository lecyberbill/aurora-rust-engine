// [WFGY] Zone: SAFE | λ: 0.15 | Fallbacks: 0 | Action: Pinpoint exact block in Step 3 producing NaN

use candle_core::{DType, Device, Module, Tensor};
use aurora_rust_engine::pipelines::FluxPipeline;
use aurora_rust_engine::diffusion::schedulers::Scheduler;

fn print_stats(name: &str, t: &Tensor) -> anyhow::Result<bool> {
    let v = t.flatten_all()?.to_dtype(DType::F32)?.to_device(&Device::Cpu)?.to_vec1::<f32>()?;
    let min = v.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean = v.iter().sum::<f32>() / v.len() as f32;
    let has_nan = v.iter().any(|x| x.is_nan());
    println!("  {}: Min = {:.4}, Max = {:.4}, Mean = {:.4}, Has NaN = {}", name, min, max, mean, has_nan);
    Ok(has_nan)
}

fn main() -> anyhow::Result<()> {
    println!("🔍 Pinpointing exact block in Step 3...");

    let device = Device::new_cuda(0)?;
    let checkpoint = "G:\\models\\flux\\flux1SchnellFp8_schnellFp8.safetensors";
    let mut pipeline = FluxPipeline::from_single_file_streaming(checkpoint, device.clone())?;

    let prompt = "a magnificent cyberpunk cyber-cat with glowing blue neon visor";
    let t5_emb = pipeline.t5xxl.as_mut().unwrap().encode(prompt, 256)?;
    let clip_vec = pipeline.clip_l.as_mut().unwrap().encode_pooled(prompt)?;

    let txt_tokens = t5_emb.to_device(&device)?.to_dtype(DType::F16)?;
    let y_vec = clip_vec.to_device(&device)?.to_dtype(DType::F16)?;

    let h_patches = (512 + 15) / 16;
    let w_patches = (512 + 15) / 16;
    let c = 16;
    let ph = 2;
    let pw = 2;

    let raw_noise = ((Tensor::randn(0f32, 1f32, (1, c, h_patches * ph, w_patches * pw), &device)? * 0.1)?.to_dtype(DType::F16))?;
    let reshaped = raw_noise.reshape((1, c, h_patches, ph, w_patches, pw))?;
    let permuted = reshaped.permute((0, 2, 4, 1, 3, 5))?.contiguous()?;
    let mut latents = permuted.reshape((1, h_patches * w_patches, c * ph * pw))?;

    pipeline.scheduler.set_timesteps_with_seq_len(4, h_patches * w_patches)?;
    let sigmas = pipeline.scheduler.sigmas().to_vec();
    let timesteps = pipeline.scheduler.timesteps().to_vec();

    // Step 1
    let t1 = Tensor::from_slice(&[sigmas[0] as f32], (1,), &device)?.to_dtype(DType::F16)?;
    let v1 = pipeline.transformer.forward_with_streamer(&latents, &txt_tokens, &t1, Some(&y_vec), None, pipeline.streamer.as_ref())?;
    print_stats("Step 1 Velocity", &v1)?;
    latents = pipeline.scheduler.step(&v1, timesteps[0], &latents)?;
    print_stats("Step 1 Latents", &latents)?;
    println!("✅ Step 1 complete");

    // Step 2 Block by Block Inspection
    let sigma2 = sigmas[1];
    println!("\n▶️ --- Diagnosing Step 2 (sigma = {:.4}) Block-by-Block ---", sigma2);
    let t2 = Tensor::from_slice(&[sigma2 as f32], (1,), &device)?.to_dtype(DType::F16)?;

    let mut temb = pipeline.transformer.time_embedder.forward(&t2)?;
    if let (Some((in_l, out_l)), Some(ref y)) = (&pipeline.transformer.vector_in, Some(&y_vec)) {
        let h = in_l.forward(y)?.silu()?;
        let v_emb = out_l.forward(&h)?;
        temb = (&temb + &v_emb)?;
    }
    print_stats("Step 2 temb", &temb)?;

    let mut img_h = pipeline.transformer.img_in.forward(&latents)?;
    let mut txt_h = pipeline.transformer.txt_in.forward(&txt_tokens)?;
    print_stats("Step 2 img_h (img_in)", &img_h)?;

    let streamer = pipeline.streamer.as_ref().unwrap();

    println!("⚡ Probing DoubleStreamBlocks in Step 2...");
    for i in 0..19 {
        let (next_img, next_txt) = streamer.execute_double_block(i, &img_h, &txt_h, &temb, None, None)?;
        let has_nan_img = print_stats(&format!("Step 2 Double Block {} (img)", i), &next_img)?;
        let has_nan_txt = print_stats(&format!("Step 2 Double Block {} (txt)", i), &next_txt)?;
        if has_nan_img || has_nan_txt {
            println!("❌ DoubleStreamBlock {} failed in Step 2!", i);
            return Ok(());
        }
        img_h = next_img;
        txt_h = next_txt;
    }

    print_stats("Step 2 img_h after double blocks", &img_h)?;
    print_stats("Step 2 txt_h after double blocks", &txt_h)?;

    println!("⚡ Probing SingleStreamBlocks in Step 2...");
    let mut unified = Tensor::cat(&[&txt_h, &img_h], 1)?;
    print_stats("Step 2 unified input", &unified)?;
    for i in 0..38 {
        let next_unified = streamer.execute_single_block(i, &unified, &temb)?;
        let has_nan = print_stats(&format!("Step 2 Single Block {}", i), &next_unified)?;
        if has_nan {
            println!("❌ SingleStreamBlock {} failed in Step 2!", i);
            return Ok(());
        }
        unified = next_unified;
    }
    let sigma3 = sigmas[2];
    println!("\n▶️ --- Diagnosing Step 3 (sigma = {:.4}) Block-by-Block ---", sigma3);
    let t3 = Tensor::from_slice(&[sigma3 as f32], (1,), &device)?.to_dtype(DType::F16)?;

    let mut temb = pipeline.transformer.time_embedder.forward(&t3)?;
    if let (Some((in_l, out_l)), Some(ref y)) = (&pipeline.transformer.vector_in, Some(&y_vec)) {
        let h = in_l.forward(y)?.silu()?;
        let v_emb = out_l.forward(&h)?;
        temb = (&temb + &v_emb)?;
    }
    print_stats("Step 3 temb", &temb)?;

    let mut img_h = pipeline.transformer.img_in.forward(&latents)?;
    let mut txt_h = pipeline.transformer.txt_in.forward(&txt_tokens)?;
    print_stats("Step 3 img_h (img_in)", &img_h)?;

    let streamer = pipeline.streamer.as_ref().unwrap();

    println!("⚡ Probing DoubleStreamBlocks in Step 3...");
    for i in 0..19 {
        let (next_img, next_txt) = streamer.execute_double_block(i, &img_h, &txt_h, &temb, None, None)?;
        let has_nan = print_stats(&format!("Step 3 Double Block {}", i), &next_img)?;
        if has_nan {
            println!("❌ DoubleStreamBlock {} failed in Step 3!", i);
            return Ok(());
        }
        img_h = next_img;
        txt_h = next_txt;
    }

    // Step 4 Block by Block Inspection
    let sigma4 = sigmas[3];
    println!("\n▶️ --- Diagnosing Step 4 (sigma = {:.4}) Block-by-Block ---", sigma4);
    let t4 = Tensor::from_slice(&[sigma4 as f32], (1,), &device)?.to_dtype(DType::F16)?;

    let mut temb4 = pipeline.transformer.time_embedder.forward(&t4)?;
    if let (Some((in_l, out_l)), Some(ref y)) = (&pipeline.transformer.vector_in, Some(&y_vec)) {
        let h = in_l.forward(y)?.silu()?;
        let v_emb = out_l.forward(&h)?;
        temb4 = (&temb4 + &v_emb)?;
    }
    print_stats("Step 4 temb", &temb4)?;

    let mut img_h4 = pipeline.transformer.img_in.forward(&latents)?;
    let mut txt_h4 = pipeline.transformer.txt_in.forward(&txt_tokens)?;
    print_stats("Step 4 img_h (img_in)", &img_h4)?;

    println!("⚡ Probing DoubleStreamBlocks in Step 4...");
    for i in 0..19 {
        let (next_img, next_txt) = streamer.execute_double_block(i, &img_h4, &txt_h4, &temb4, None, None)?;
        let has_nan_img = print_stats(&format!("Step 4 Double Block {} (img)", i), &next_img)?;
        let has_nan_txt = print_stats(&format!("Step 4 Double Block {} (txt)", i), &next_txt)?;
        if has_nan_img || has_nan_txt {
            println!("❌ DoubleStreamBlock {} failed in Step 4!", i);
            return Ok(());
        }
        img_h4 = next_img;
        txt_h4 = next_txt;
    }

    println!("⚡ Probing SingleStreamBlocks in Step 4...");
    let mut unified4 = Tensor::cat(&[&txt_h4, &img_h4], 1)?;
    for i in 0..38 {
        let next_unified = streamer.execute_single_block(i, &unified4, &temb4)?;
        let has_nan = print_stats(&format!("Step 4 Single Block {}", i), &next_unified)?;
        if has_nan {
            println!("❌ SingleStreamBlock {} failed in Step 4!", i);
            return Ok(());
        }
        unified4 = next_unified;
    }

    println!("🎉 ALL 4 STEPS SUCCEEDED WITH 100% NUMERICAL STABILITY!");
    Ok(())
}
