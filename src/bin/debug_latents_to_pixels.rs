// [WFGY] Zone: SAFE | λ: 0.15 | Fallbacks: 0 | Action: Debug latents values and RGB image creation

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

fn unpatchify(latents: &Tensor, height: usize, width: usize) -> candle_core::Result<Tensor> {
    let h_patches = (height + 15) / 16;
    let w_patches = (width + 15) / 16;
    let c = 16;
    let ph = 2;
    let pw = 2;

    let reshaped = latents.reshape((1, h_patches, w_patches, c, ph, pw))?;
    let permuted = reshaped.permute((0, 3, 1, 4, 2, 5))?.contiguous()?;
    permuted.reshape((1, c, h_patches * ph, w_patches * pw))
}

fn main() -> anyhow::Result<()> {
    println!("🔍 Debugging Latents to VAE to Pixel conversion...");

    let device = Device::new_cuda(0)?;
    let checkpoint = "G:\\models\\flux\\flux1SchnellFp8_schnellFp8.safetensors";
    let mut pipeline = FluxPipeline::from_single_file_streaming(checkpoint, device.clone())?;

    let prompt = "a majestic cat";
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

    for (step_idx, &t) in timesteps.iter().enumerate() {
        let sigma = sigmas[step_idx];
        let t_tensor = Tensor::from_slice(&[sigma as f32], (1,), &device)?.to_dtype(DType::F16)?;
        let velocity = pipeline.transformer.forward_with_streamer(&latents, &txt_tokens, &t_tensor, Some(&y_vec), None, pipeline.streamer.as_ref())?;
        print_stats(&format!("Step {} Velocity", step_idx + 1), &velocity)?;
        latents = pipeline.scheduler.step(&velocity, t, &latents)?;
        print_stats(&format!("Step {} Latents", step_idx + 1), &latents)?;
    }

    println!("\n📦 Testing Unpatchify...");
    let unpatch = unpatchify(&latents, 512, 512)?;
    print_stats("Unpatchified Latents", &unpatch)?;

    let vae = pipeline.vae.as_ref().unwrap();
    println!("\n✨ Testing VAE Raw Decode...");
    let decoded_rgb = vae.decode(&unpatch)?;
    print_stats("Decoded RGB Raw", &decoded_rgb)?;

    let img = vae.decode_to_image(&unpatch)?;
    let raw_bytes = img.as_raw();
    println!("🎨 Image Bytes Stats: Min = {}, Max = {}, Len = {}", raw_bytes.iter().min().unwrap(), raw_bytes.iter().max().unwrap(), raw_bytes.len());

    Ok(())
}
