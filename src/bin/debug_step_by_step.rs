// [WFGY] Zone: SAFE | λ: 0.15 | Fallbacks: 0 | Action: Debug Step 1 -> Step 2 ODE transition to inspect where NaN appears

use candle_core::{DType, Device, Tensor};
use aurora_rust_engine::pipelines::FluxPipeline;
use aurora_rust_engine::diffusion::schedulers::Scheduler;

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
    println!("🔍 Inspecting Step 1 -> Step 2 ODE Step...");

    let device = Device::new_cuda(0)?;
    let checkpoint = "G:\\models\\flux\\flux1SchnellFp8_schnellFp8.safetensors";
    let mut pipeline = FluxPipeline::from_single_file_streaming(checkpoint, device.clone())?;

    let prompt = "a magnificent cyberpunk cyber-cat with glowing blue neon visor";
    let t5_emb = pipeline.t5xxl.as_mut().unwrap().encode(prompt, 256)?;
    let clip_vec = pipeline.clip_l.as_mut().unwrap().encode_pooled(prompt)?;

    let txt_tokens = t5_emb.to_device(&device)?.to_dtype(DType::F16)?;
    let y_vec = clip_vec.to_device(&device)?.to_dtype(DType::F16)?;

    // Latents initialization exactly as in pipeline
    let h_patches = (512 + 15) / 16;
    let w_patches = (512 + 15) / 16;
    let c = 16;
    let ph = 2;
    let pw = 2;

    let raw_noise = ((Tensor::randn(0f32, 1f32, (1, c, h_patches * ph, w_patches * pw), &device)? * 0.1)?.to_dtype(DType::F16))?;
    print_stats("0. raw_noise", &raw_noise)?;

    let reshaped = raw_noise.reshape((1, c, h_patches, ph, w_patches, pw))?;
    let permuted = reshaped.permute((0, 2, 4, 1, 3, 5))?.contiguous()?;
    let mut latents = permuted.reshape((1, h_patches * w_patches, c * ph * pw))?;
    print_stats("1. Initial packed latents", &latents)?;

    pipeline.scheduler.set_timesteps(4)?;
    let sigmas = pipeline.scheduler.sigmas().to_vec();
    let timesteps = pipeline.scheduler.timesteps().to_vec();

    for (step_idx, &t) in timesteps.iter().enumerate() {
        let sigma = sigmas[step_idx];
        println!("\n▶️ --- Step {} (t={}, sigma={:.4}) ---", step_idx + 1, t, sigma);

        let t_tensor = Tensor::from_slice(&[sigma as f32], (1,), &device)?.to_dtype(DType::F16)?;
        
        let velocity = pipeline.transformer.forward_with_streamer(
            &latents,
            &txt_tokens,
            &t_tensor,
            Some(&y_vec),
            None,
            pipeline.streamer.as_ref(),
        )?;
        print_stats(&format!("Step {} Velocity", step_idx + 1), &velocity)?;

        latents = pipeline.scheduler.step(&velocity, t, &latents)?;
        print_stats(&format!("Step {} Latents after ODE", step_idx + 1), &latents)?;
    }

    Ok(())
}
