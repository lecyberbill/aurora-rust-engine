// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: Verification of Flux.2-Dev (guidance) Streaming Inference

use candle_core::{DType, Device, Result};
use std::path::{Path, PathBuf};
use std::time::Instant;
use aurora_rust_engine::pipelines::FluxPipeline;
use aurora_rust_engine::traits::DiffusionParams;
use aurora_rust_engine::weights::{SafeTensorsArchive, WeightRouter};
use aurora_rust_engine::diffusion::vae_flux::FluxVaeDecoder;

fn main() -> Result<()> {
    println!("================================================================================");
    println!("🚀 AURORA PURE RUST FLUX.2-DEV STREAMING VERIFICATION");
    println!("   • 6144 Hidden, 8 Double Blocks, 48 Single Blocks, Guidance Embed");
    println!("================================================================================");

    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    let checkpoint = "G:\\models\\flux\\flux2DevFp8Scaled_fp8Scaled.safetensors";
    let mistral_path = "G:\\models\\clip\\mistral_3_small_flux2_fp8.safetensors";
    let vae_path = "G:\\models\\vae\\flux2-vae.safetensors";

    if !Path::new(checkpoint).exists() {
        eprintln!("[-] Checkpoint not found: {}", checkpoint);
        return Ok(());
    }

    println!("\n📥 Loading Flux.2-Dev Checkpoint with Sequential Streamer...");
    let t_start = Instant::now();
    let mut pipeline = FluxPipeline::from_single_file_streaming(checkpoint, device.clone())
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    pipeline.enable_flash_attn();

    if Path::new(mistral_path).exists() {
        println!("📥 Attaching Mistral-3-Small Prompt Encoder (CPU, FP8 dequant)...");
        let mistral = aurora_rust_engine::text::Mistral3TextEncoder::from_safetensors(
            mistral_path,
            Some(std::path::Path::new("mistral_tokenizer.json")),
            Device::Cpu,
            DType::F16,
        ).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        pipeline.set_mistral(mistral);
        println!("✅ Mistral-3-Small Attached!");
    }

    if Path::new(vae_path).exists() {
        println!("📥 Attaching Flux.2 32-Channel VAE Decoder (GPU/F16)...");
        let vae_archive = SafeTensorsArchive::open(PathBuf::from(vae_path))
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let vae_router = WeightRouter::new(&vae_archive, device.clone(), DType::F16);
        let vae_vb = vae_router.vae_var_builder()
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let decoder = FluxVaeDecoder::new(vae_vb)?;
        pipeline.set_vae(decoder);
        println!("✅ Flux.2 VAE Decoder Attached!");
    }

    println!("✅ Pipeline initialization completed in {:.2}s", t_start.elapsed().as_secs_f64());

    let prompt = "a gorgeous portrait of an arctic fox with sapphire blue eyes in a mystical snowy forest at twilight, cinematic lighting, 8k";
    let steps: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(30);
    let guidance: f64 = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(3.5);
    println!("\n🎨 Generating Image with FLUX.2-Dev ({} Steps, Guidance {})...", steps, guidance);
    println!("📝 Prompt: \"{}\"", prompt);

    let params = DiffusionParams {
        prompt,
        negative_prompt: None,
        num_steps: steps,
        guidance_scale: guidance,
        width: 1024,
        height: 1024,
        seed: 42,
    };

    let (image, metrics) = pipeline.generate_with_metrics(params, None::<fn(usize, usize, &candle_core::Tensor)>)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

    let out_path = format!("outputs/flux_showcase/flux_dev_1024_s{}_g{}.png", steps, guidance);
    image.save(&out_path)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to save output: {}", e)))?;

    println!("\n================================================================================");
    println!("🎉 FLUX.2-Dev Generation Finished Successfully!");
    println!("📁 Output Path: {}", out_path);
    println!("📊 Performance Telemetry:");
    println!("   • Active Steps:           {}", metrics.unet_steps);
    println!("   • ODE Denoising Duration: {:.2}s", metrics.unet_total_ms / 1000.0);
    println!("   • Throughput:             {:.2} it/s ({:.2} ms/step)", metrics.unet_it_per_sec, metrics.unet_step_avg_ms);
    println!("   • VAE & Unpatchify:       {:.2}s", metrics.vae_decode_ms / 1000.0);
    println!("   • Total Wall-Clock:       {:.2}s", metrics.total_wallclock_ms / 1000.0);
    println!("================================================================================\n");
    Ok(())
}
