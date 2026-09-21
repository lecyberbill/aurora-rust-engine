use std::path::PathBuf;
use candle_core::{DType, Tensor};
use aurora_rust_engine::device::auto_device;
use aurora_rust_engine::diffusion::dit::z_image::{ZImageConfig, ZImageTransformer};
use aurora_rust_engine::diffusion::schedulers::{FlowMatchEulerConfig, FlowMatchEulerScheduler, Scheduler};
use aurora_rust_engine::diffusion::vae_flux::FluxVaeDecoder;
use aurora_rust_engine::text::Qwen3TextEncoder;
use aurora_rust_engine::weights::SafeTensorsArchive;
use std::sync::Arc;

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
    println!("   📊 [{}] shape={:?} | mean={:.4} | std={:.4} | min={:.4} | max={:.4}", name, shape, mean, std, min, max);
    Ok(())
}

fn main() -> anyhow::Result<()> {
    println!("================================================================================");
    println!("🔬 Isolation Probe: Z-Image Turbo Diagnostics (Lumina2 / NextDiT)");
    println!("================================================================================");

    let device = auto_device()?;
    let dtype = if device.is_cuda() { DType::BF16 } else { DType::F32 };
    println!("🎮 Device: {:?} | DType: {:?}", device, dtype);

    let aio_path = PathBuf::from(r"G:\models\zit\z-image-turbo-fp8-aio.safetensors");
    let vae_path = PathBuf::from(r"G:\models\zit\zImageTurbo_vae.safetensors");

    let archive = Arc::new(SafeTensorsArchive::open(&aio_path)?);

    // 1. Probe Text Encoder
    println!("\n--- [Probe 1: Qwen3 Text Encoder] ---");
    let text_encoder = Qwen3TextEncoder::from_archive(archive.as_ref(), None, &device, dtype)?;
    println!("   Tokenizer status: present={}", text_encoder.has_tokenizer());
    let prompt = "A cinematic futuristic sports car driving through a neon cyber city at night, 8k octane render, hyperdetailed";
    let context = text_encoder.encode_last_hidden(prompt, 256)?;
    print_stats("Qwen3 Context Embeddings (layer 34)", &context)?;

    // 2. Probe DiT Model Architecture & Forward pass
    println!("\n--- [Probe 2: Z-Image DiT Transformer] ---");
    let mut dit_tensors = std::collections::HashMap::new();
    for key in archive.keys() {
        if let Some(rest) = key.strip_prefix("model.diffusion_model.") {
            if let Ok(t) = archive.get_tensor(&key, &device, dtype) {
                dit_tensors.insert(rest.to_string(), t);
            }
        }
    }
    let dit_vb = candle_nn::VarBuilder::from_tensors(dit_tensors, dtype, &device);
    let config = ZImageConfig::default();
    let transformer = ZImageTransformer::new(config, dit_vb)?;

    let latents = Tensor::randn(0.0f32, 1.0f32, (1, 16, 64, 64), &device)?.to_dtype(dtype)?;
    print_stats("Initial Latents x_0", &latents)?;

    let t_tensor = Tensor::from_vec(vec![0.0f32], (1,), &device)?.to_dtype(dtype)?; // sigma = 1.0 -> lumina_t = 0.0
    let pred_v = transformer.forward(&latents, &t_tensor, &context)?;
    print_stats("Predicted Velocity (t=0)", &pred_v)?;

    // 3. Probe Denoising Steps (4 steps)
    println!("\n--- [Probe 3: Flow Match Scheduler Simulation] ---");
    let mut scheduler = FlowMatchEulerScheduler::new(FlowMatchEulerConfig {
        shift: 3.0,
        base_shift: 0.5,
        max_shift: 1.15,
        min_shift: 0.5,
        use_dynamic_shifting: false,
        double_shift_linspace: false,
    });
    scheduler.set_timesteps(4)?;
    let timesteps = scheduler.timesteps().to_vec();
    let sigmas = scheduler.sigmas().to_vec();
    println!("   Timesteps: {:?}", timesteps);
    println!("   Sigmas: {:?}", sigmas);

    let mut cur_latents = latents.clone();
    for (step_idx, &t) in timesteps.iter().enumerate() {
        let sigma = t as f32 / 1000.0f32;
        let lumina_t = 1.0f32 - sigma;
        let t_t = Tensor::from_vec(vec![lumina_t], (1,), &device)?.to_dtype(dtype)?;
        let v = transformer.forward(&cur_latents, &t_t, &context)?;
        print_stats(&format!("Step {} pred_v (sigma={:.3})", step_idx + 1, sigma), &v)?;
        cur_latents = scheduler.step(&v, t, &cur_latents)?;
        print_stats(&format!("Step {} latents_next", step_idx + 1), &cur_latents)?;
    }

    // 4. Probe VAE Decoding
    println!("\n--- [Probe 4: VAE Decoding Scaling Test] ---");
    let vae_archive = Arc::new(SafeTensorsArchive::open(&vae_path)?);
    let mut vae_tensors = std::collections::HashMap::new();
    for key in vae_archive.keys() {
        if let Ok(t) = vae_archive.get_tensor(&key, &device, dtype) {
            vae_tensors.insert(key.to_string(), t);
        }
    }
    let vae_vb = candle_nn::VarBuilder::from_tensors(vae_tensors, dtype, &device);
    let vae = FluxVaeDecoder::new(vae_vb)?;

    let decoded_default = vae.decode(&cur_latents)?;
    print_stats("Decoded with Flux 1 scale ((z/0.3611)+0.1159)", &decoded_default)?;

    println!("\n✅ Isolation Probe Completed successfully.");
    Ok(())
}
