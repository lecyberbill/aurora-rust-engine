// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: Generate and save Flux.1 image to disk

use candle_core::{DType, Device};
use aurora_rust_engine::traits::DiffusionParams;
use aurora_rust_engine::pipelines::FluxPipeline;
use aurora_rust_engine::weights::{SafeTensorsArchive, WeightRouter};
use aurora_rust_engine::text::Qwen3TextEncoder;
use aurora_rust_engine::diffusion::vae_flux::FluxVaeDecoder;
use std::fs;
use std::path::Path;
use std::time::Instant;

fn progress_cb(step: usize, total: usize, _latent: &candle_core::Tensor) {
    println!("   [Step {}/{}] Flow Match ODE Integrated", step, total);
}

fn main() -> anyhow::Result<()> {
    println!("================================================================================");
    println!("🚀 AURORA PURE RUST FLUX.1 SCHNELL / DEV IMAGE GENERATION");
    println!("   • MMDiT Multimodal Diffusion Transformer");
    println!("   • Flow Matching Euler ODE Solver (4 Steps)");
    println!("   • 2D Unpatchify + Latent Decoded Render");
    println!("================================================================================\n");

    let output_dir = "outputs/flux_showcase";
    fs::create_dir_all(output_dir)?;

    let args: Vec<String> = std::env::args().collect();
    let flux_checkpoint = if args.len() > 1 {
        args[1].clone()
    } else {
        "G:\\models\\flux\\flux1-dev-fp8.safetensors".to_string()
    };

    let is_dev = flux_checkpoint.to_lowercase().contains("dev");
    let num_steps = if args.len() > 2 {
        args[2].parse::<usize>().unwrap_or(if is_dev { 20 } else { 4 })
    } else {
        if is_dev { 20 } else { 4 }
    };
    let guidance_scale = if is_dev { 3.5 } else { 1.0 };

    if !Path::new(&flux_checkpoint).exists() {
        eprintln!("[-] Flux checkpoint not found at: {}", flux_checkpoint);
        return Ok(());
    }

    let device = Device::new_cuda(0)?;
    println!("📥 Loading Flux Checkpoint: {}", flux_checkpoint);
    let t_load = Instant::now();
    let mut pipeline = FluxPipeline::from_single_file(&flux_checkpoint, device.clone())?;

    let is_klein = flux_checkpoint.to_lowercase().contains("klein");
    if is_klein {
        let qwen_path = "G:\\models\\clip\\qwen_3_4b.safetensors";
        if Path::new(qwen_path).exists() {
            println!("📥 Loading External Qwen3-4B Text Encoder: {}", qwen_path);
            let qwen_archive = SafeTensorsArchive::open(qwen_path)?;
            let mut tensors = std::collections::HashMap::new();
            for k in qwen_archive.keys() {
                if let Ok(t) = qwen_archive.get_tensor(k, &Device::Cpu, DType::F32) {
                    tensors.insert(k.to_string(), t);
                }
            }
            let qwen_vb = candle_nn::VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
            let qwen = Qwen3TextEncoder::new(qwen_vb, None)?;
            pipeline.set_qwen3(qwen);
            println!("✅ Qwen3-4B Text Encoder Attached!");
        }

        let vae_path = "G:\\models\\vae\\flux2-vae.safetensors";
        if Path::new(vae_path).exists() {
            println!("📥 Loading External Flux.2 32-Channel VAE: {}", vae_path);
            let vae_archive = SafeTensorsArchive::open(vae_path)?;
            let vae_router = WeightRouter::new(&vae_archive, device.clone(), DType::F16);
            let vae_vb = vae_router.vae_var_builder()?;
            let vae = FluxVaeDecoder::new(vae_vb)?;
            pipeline.set_vae(vae);
            println!("✅ Flux.2 VAE Decoder Attached!");
        }
    }
    println!("✅ Flux Pipeline loaded in {:.2}s\n", t_load.elapsed().as_secs_f64());

    let default_prompt = if is_klein {
        "a steaming cup of hot coffee on a wooden table, morning light, rich aroma, cinematic lighting, masterpiece, photorealistic, 8k"
    } else {
        "masterpiece, highly detailed, futuristic glowing robot, neon lights, 8k"
    };
    let prompt_str = if args.len() > 3 {
        args[3].clone()
    } else {
        default_prompt.to_string()
    };

    let params = DiffusionParams {
        prompt: &prompt_str,
        negative_prompt: None,
        num_steps,
        guidance_scale,
        width: 1024,
        height: 1024,
        seed: 42,
    };

    println!("⚡ Executing {}-Step Flow Matching ODE Generation (Guidance: {:.1})...", num_steps, guidance_scale);
    let (img_out, metrics) = pipeline.generate_with_metrics(params, Some(progress_cb))?;

    let model_tag = if is_klein {
        "klein_4b"
    } else if is_dev {
        "dev"
    } else {
        "schnell"
    };
    let file_name = format!("{}/flux_{}_1024_seed42.png", output_dir, model_tag);
    img_out.save(&file_name)?;

    println!("\n================================================================================");
    println!("🎉 Flux.1 Image Generated & Saved Successfully!");
    println!("📁 Output Path: {}", file_name);
    println!("📊 Flux.1 Telemetry Metrics:");
    println!("   • ODE Denoising Duration: {:.2}s", metrics.unet_total_ms / 1000.0);
    println!("   • Throughput:             {:.2} it/s ({:.2} ms/step)", metrics.unet_it_per_sec, metrics.unet_step_avg_ms);
    println!("   • VAE & Unpatchify:       {:.2}s", metrics.vae_decode_ms / 1000.0);
    println!("   • Total Wall-Clock:       {:.2}s", metrics.total_wallclock_ms / 1000.0);
    println!("================================================================================\n");

    Ok(())
}
