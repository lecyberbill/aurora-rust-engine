// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: Pure Rust Krea 2 Turbo Step Diagnostic Probe

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use candle_core::{DType, Device, Tensor};
use aurora_rust_engine::device::auto_device;
use aurora_rust_engine::diffusion::dit::z_image::{SingleStreamBlock, ZImageConfig, ZImageTransformer};
use aurora_rust_engine::diffusion::schedulers::{FlowMatchEulerConfig, FlowMatchEulerScheduler, Scheduler};
use aurora_rust_engine::diffusion::vae_qwen::QwenImageVaeDecoder;
use aurora_rust_engine::text::Qwen3TextEncoder;
use aurora_rust_engine::weights::SafeTensorsArchive;

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
    println!("   📊 [{:<25}] shape={:?} | mean={:+.4} | std={:.4} | min={:+.4} | max={:+.4}", name, shape, mean, std, min, max);
    Ok(())
}

fn main() -> anyhow::Result<()> {
    println!("================================================================================");
    println!("⚡ Aurora Engine: Krea 2 Turbo Isolated Step Diagnostic Probe");
    println!("================================================================================");

    #[cfg(feature = "rocm")]
    let device = match Device::new_rocm(0) {
        Ok(d) => {
            println!("🚀 ROCm GPU device 0 successfully initialized!");
            d
        }
        Err(e) => {
            println!("⚠️ ROCm init error: {:?}. Falling back to auto_device.", e);
            auto_device()?
        }
    };
    #[cfg(not(feature = "rocm"))]
    let device = auto_device()?;

    let dtype = if !device.is_cpu() { DType::BF16 } else { DType::F32 };
    println!("🎮 Target Device: {:?} | Precision: {:?}", device, dtype);

    let dit_path = PathBuf::from(std::env::args().nth(1).unwrap_or_else(|| "/models/comfyui/diffusion_models/krea2_turbo_fp8_scaled.safetensors".to_string()));
    let te_path = PathBuf::from(std::env::args().nth(2).unwrap_or_else(|| "/models/comfyui/text_encoders/qwen3vl_4b_fp8_scaled.safetensors".to_string()));
    let vae_path = PathBuf::from(std::env::args().nth(3).unwrap_or_else(|| "/models/comfyui/vae/qwen_image_vae_complet.safetensors".to_string()));

    println!("📂 Loading Text Encoder from {:?}", te_path);
    let te_archive = SafeTensorsArchive::open(&te_path)?;
    let tok_path = PathBuf::from("qwen_tokenizer_KREA2.json");
    let text_encoder = Qwen3TextEncoder::from_archive(&te_archive, Some(&tok_path), &device, dtype)?;
    println!("🔍 Text Encoder has tokenizer: {}", text_encoder.has_tokenizer());

    let prompt = "a majestic white wolf on a snowy cliff at golden hour, photorealistic, 8k";
    println!("🧠 Encoding prompt: '{}'", prompt);
    let context = text_encoder.encode_krea2_12_layers(prompt, 0)?
        .to_device(&device)?
        .to_dtype(dtype)?;
    print_stats("Qwen3 Taps Context (Cond)", &context)?;

    let uncond_context = text_encoder.encode_krea2_12_layers("", 0)?
        .to_device(&device)?
        .to_dtype(dtype)?;
    print_stats("Qwen3 Taps Context (Uncond)", &uncond_context)?;

    let cfg_scale: f64 = std::env::var("KREA_CFG").ok().and_then(|s| s.parse().ok()).unwrap_or(1.0);
    println!("🎛️ CFG Guidance Scale: {:.2}", cfg_scale);

    println!("\n📂 Loading DiT Transformer in memory from {:?}", dit_path);
    let dit_archive = Arc::new(SafeTensorsArchive::open(&dit_path)?);
    let keys: Vec<String> = dit_archive.keys().cloned().collect();

    println!("🔑 Inspecting Safetensors Archive Weights (sample stats):");
    for k in &["first.weight", "first.bias", "last.linear.weight", "last.linear.bias", "last.norm.scale", "last.modulation.lin", "tproj.1.weight", "tproj.1.bias", "blocks.0.mod.lin", "blocks.0.prenorm.scale", "blocks.0.attn.wo.weight", "blocks.0.attn.qknorm.qnorm.scale", "blocks.0.attn.qknorm.knorm.scale"] {
        if let Ok(t) = dit_archive.get_tensor(k, &Device::Cpu, DType::F32) {
            print_stats(k, &t)?;
        }
    }

    let mut header_tensors = std::collections::HashMap::new();
    let mut probe_tensors = std::collections::HashMap::new();
    for key in &keys {
        if key.ends_with(".weight_scale") || key.ends_with(".scale_weight") || key.ends_with(".comfy_quant") {
            continue;
        }
        let is_header = key.starts_with("txtfusion.")
            || key.starts_with("first.")
            || key.starts_with("last.")
            || key.starts_with("tmlp.")
            || key.starts_with("tproj.")
            || key.starts_with("txtmlp.")
            || key.starts_with("model.diffusion_model.txtfusion.")
            || key.starts_with("model.diffusion_model.first.")
            || key.starts_with("model.diffusion_model.last.")
            || key.starts_with("model.diffusion_model.tmlp.")
            || key.starts_with("model.diffusion_model.tproj.")
            || key.starts_with("model.diffusion_model.txtmlp.");

        let is_probe = key.starts_with("blocks.0.") || key.starts_with("model.diffusion_model.blocks.0.");
        let rest = key.strip_prefix("model.diffusion_model.")
            .or_else(|| key.strip_prefix("diffusion_model."))
            .unwrap_or(key);

        if is_header {
            if let Ok(t) = dit_archive.get_tensor(key, &device, dtype) {
                header_tensors.insert(rest.to_string(), t);
            }
        } else if is_probe {
            if let Ok(t) = dit_archive.get_tensor(key, &Device::Cpu, DType::F32) {
                probe_tensors.insert(rest.to_string(), t);
            }
        }
    }

    let mut config_map = std::collections::HashMap::new();
    for (k, v) in &header_tensors { config_map.insert(k.clone(), v.clone()); }
    for (k, v) in probe_tensors { config_map.insert(k, v); }
    let mut config = ZImageConfig::from_tensors_and_keys(&config_map, &keys);
    if let Ok(theta_str) = std::env::var("KREA_THETA") {
        if let Ok(th) = theta_str.parse::<f64>() {
            config.theta = th;
        }
    }
    println!("⚙️ DiT Architecture: hidden={}, heads={}, layers={}, in_ch={}, theta={}",
        config.hidden_size, config.num_heads, config.num_layers, config.in_channels, config.theta);

    let header_vb = candle_nn::VarBuilder::from_tensors(header_tensors, dtype, &device);
    let transformer = ZImageTransformer::new_streaming(config, header_vb, dit_archive.clone(), device.clone(), dtype)?;

    // Setup Flow Match Scheduler with official Krea 2 schedule formula:
    // ts_lin = linspace(1, 0, steps+1)
    // For Raw 1024x1024 (N_img = 4096): mu = 0.90625
    // ts = exp(mu) / (exp(mu) + (1/ts_lin - 1))
    let num_steps: usize = std::env::var("KREA_STEPS").ok().and_then(|s| s.parse().ok()).unwrap_or(8);
    let mu: f64 = std::env::var("KREA_MU").ok().and_then(|s| s.parse().ok()).unwrap_or(1.15);
    let guidance: f64 = std::env::var("KREA_GUIDANCE").ok().and_then(|s| s.parse().ok()).unwrap_or(0.0);

    println!("🎛️ Inference Mode: Krea 2 Turbo (Steps={}, mu={:.4}, Guidance={:.2})", num_steps, mu, guidance);

    let mut ts_vec = Vec::with_capacity(num_steps + 1);
    let exp_mu = mu.exp();
    for i in 0..=num_steps {
        let ts_lin = 1.0 - (i as f64) / (num_steps as f64);
        if ts_lin <= 0.0 {
            ts_vec.push(0.0f64);
        } else if ts_lin >= 1.0 {
            ts_vec.push(1.0f64);
        } else {
            let val = exp_mu / (exp_mu + (1.0 / ts_lin - 1.0));
            ts_vec.push(val);
        }
    }
    println!("📋 Timesteps ts: {:?}", ts_vec);

    // Initial noise latent (seed 42 deterministic)
    let b = 1;
    let latent_h = 128;
    let latent_w = 128;
    let mut latents = Tensor::randn(0.0f32, 1.0f32, (b, 16, latent_h, latent_w), &device)?.to_dtype(dtype)?;
    print_stats("Initial Noise x_0", &latents)?;

    println!("\n📂 Pre-loading VAE decoder:");
    let vae_archive = SafeTensorsArchive::open(&vae_path)?;
    let mut vae_tensors = std::collections::HashMap::new();
    for k in vae_archive.keys() {
        if let Ok(t) = vae_archive.get_tensor(&k, &Device::Cpu, DType::F32) {
            vae_tensors.insert(k.to_string(), t);
        }
    }
    let vae_vb = candle_nn::VarBuilder::from_tensors(vae_tensors, DType::F32, &Device::Cpu);
    let vae = QwenImageVaeDecoder::new(vae_vb)?;

    std::fs::create_dir_all("outputs")?;

    // Euler loop matching official Krea 2 formula:
    // for (t_cur, t_next) in zip(ts[:-1], ts[1:]):
    //     v = cond + guidance * (cond - uncond) if guidance > 0 else cond
    //     x = x + (t_next - t_cur) * v
    for step_idx in 0..num_steps {
        let t_cur = ts_vec[step_idx];
        let t_next = ts_vec[step_idx + 1];
        let dt = (t_next - t_cur) as f32; // negative step

        let t_tensor = Tensor::from_vec(vec![t_cur as f32], (1,), &device)?.to_dtype(dtype)?;
        println!("\n🔄 Running Step {}/{} (t_cur={:.4} -> t_next={:.4}, dt={:.4}):", step_idx + 1, num_steps, t_cur, t_next, dt);
        let t0 = Instant::now();
        
        let pred_v = if guidance > 0.0 {
            let v_cond = transformer.forward(&latents, &t_tensor, &context)?;
            let v_uncond = transformer.forward(&latents, &t_tensor, &uncond_context)?;
            let diff = (&v_cond - &v_uncond)?;
            (&v_cond + (diff * guidance)?)?
        } else {
            transformer.forward(&latents, &t_tensor, &context)?
        };
        print_stats(&format!("Step {} pred_v", step_idx + 1), &pred_v)?;

        // x_0 direct estimation: x_0 = x_t - t_cur * v
        let x0_est = (&latents - (&pred_v * t_cur)?)?;
        let cpu_x0 = x0_est.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
        if (step_idx + 1) % 4 == 0 || step_idx + 1 == num_steps || step_idx == 0 {
            if let Ok(decoded_x0) = vae.decode(&cpu_x0) {
                if let Ok(img) = aurora_rust_engine::diffusion::vae::tensor_to_rgb_image(&decoded_x0) {
                    let path = format!("outputs/probe_x0_step{}.png", step_idx + 1);
                    img.save(&path)?;
                    println!("   📸 Saved direct x0 estimate to {}", path);
                }
            }
        }

        // Standard FlowMatch Euler step: x_next = x_cur + dt * v (dt < 0)
        let lat_next = (&latents + (&pred_v * (dt as f64))?)?;
        print_stats(&format!("Step {} lat_next", step_idx + 1), &lat_next)?;

        latents = lat_next;
        println!("   Elapsed: {:.2}s", t0.elapsed().as_secs_f64());
    }

    println!("\n📂 Final decode:");
    let cpu_latents = latents.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    let decoded = vae.decode(&cpu_latents)?;
    print_stats("Decoded RGB Tensor", &decoded)?;

    let img_std = aurora_rust_engine::diffusion::vae::tensor_to_rgb_image(&decoded)?;
    img_std.save("outputs/probe_step8.png")?;
    img_std.save("outputs/probe_step8_std.png")?;

    let min_val = decoded.flatten_all()?.to_vec1::<f32>()?.into_iter().fold(f32::INFINITY, f32::min);
    let max_val = decoded.flatten_all()?.to_vec1::<f32>()?.into_iter().fold(f32::NEG_INFINITY, f32::max);
    let range = (max_val - min_val).max(1e-5);
    let norm_stretch = (((&decoded - min_val as f64)? / range as f64)? * 255.0)?.clamp(0.0f32, 255.0f32)?.to_dtype(DType::U8)?;
    let hw3_stretch = norm_stretch.squeeze(0)?.permute((1, 2, 0))?.contiguous()?;
    let flat_stretch = hw3_stretch.flatten_all()?.to_vec1::<u8>()?;
    if let Some(img_stretch) = image::RgbImage::from_raw(1024, 1024, flat_stretch) {
        img_stretch.save("outputs/probe_step8_autocontrast.png")?;
    }

    println!("🎉 Saved images to outputs/probe_step8.png, probe_step8_direct.png, probe_step8_autocontrast.png");
    Ok(())
}

