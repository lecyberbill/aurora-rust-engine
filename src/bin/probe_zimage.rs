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

    let aio_path = std::env::args().nth(1)
        .map(PathBuf::from)
        .or_else(|| std::env::var("MODEL_PATH").ok().map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from(r"G:\models\zit\z-image-turbo-fp8-aio.safetensors"));
    let vae_path = std::env::args().nth(2)
        .map(PathBuf::from)
        .or_else(|| std::env::var("VAE_PATH").ok().map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from(r"/models/comfyui/vae/qwen_image_vae.safetensors"));

    println!("📂 Opening: {:?}", aio_path);
    let archive = Arc::new(SafeTensorsArchive::open(&aio_path)?);
    let keys: Vec<String> = archive.keys().map(|k| k.to_string()).collect();
    println!("🔑 Total keys: {}", keys.len());
    println!("📋 Non-block keys in archive:");
    for k in keys.iter().filter(|k| !k.starts_with("blocks.") && !k.starts_with("model.diffusion_model.blocks.")) {
        println!("   • {}", k);
    }

    // 1. Probe Text Encoder (if embedded)
    println!("\n--- [Probe 1: Qwen3 Text Encoder] ---");
    let context = match Qwen3TextEncoder::from_archive(archive.as_ref(), None, &device, dtype) {
        Ok(text_encoder) => {
            println!("   Tokenizer status: present={}", text_encoder.has_tokenizer());
            let prompt = "A cinematic futuristic sports car driving through a neon cyber city at night, 8k octane render, hyperdetailed";
            let ctx = text_encoder.encode_last_hidden(prompt, 256)?;
            print_stats("Qwen3 Context Embeddings", &ctx)?;
            ctx
        }
        Err(e) => {
            println!("   ℹ️ Text encoder not in this checkpoint ({:?}), creating dummy context for DiT probe", e);
            Tensor::zeros((1, 16, 2560), dtype, &device)?
        }
    };

    // 2. Probe DiT Model Architecture & Forward pass
    println!("\n--- [Probe 2: Z-Image DiT Transformer] ---");
    let mut dit_tensors = std::collections::HashMap::new();
    for key in archive.keys() {
        let rest = key.strip_prefix("model.diffusion_model.")
            .or_else(|| key.strip_prefix("diffusion_model."))
            .unwrap_or(&key);
        if let Ok(t) = archive.get_tensor(&key, &device, dtype) {
            dit_tensors.insert(rest.to_string(), t);
        }
    }
    println!("   Found {} DiT tensors in archive", dit_tensors.len());
    let dit_vb = candle_nn::VarBuilder::from_tensors(dit_tensors, dtype, &device);
    let config = ZImageConfig::default();
    let transformer = match ZImageTransformer::new(config, dit_vb) {
        Ok(t) => {
            println!("   ✅ ZImageTransformer successfully instantiated!");
            Some(t)
        }
        Err(e) => {
            println!("   ❌ ZImageTransformer instantiation error: {:?}", e);
            None
        }
    };

    if let Some(ref transformer) = transformer {
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
    }

    println!("\n✅ Isolation Probe Completed successfully.");
    Ok(())
}
