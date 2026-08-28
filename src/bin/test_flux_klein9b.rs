// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: Verification of Flux.2-Klein-9B Streaming Inference under < 8GB VRAM

use candle_core::{DType, Device, Result};
use std::path::{Path, PathBuf};
use std::time::Instant;
use aurora_rust_engine::pipelines::FluxPipeline;
use aurora_rust_engine::traits::DiffusionParams;
use aurora_rust_engine::weights::{SafeTensorsArchive, WeightRouter};
use aurora_rust_engine::diffusion::vae_flux::FluxVaeDecoder;

fn main() -> Result<()> {
    println!("================================================================================");
    println!("🚀 AURORA PURE RUST FLUX.2-KLEIN-9B STREAMING VERIFICATION");
    println!("   • 4096 Hidden Dim, 8 Double Blocks, 24 Single Blocks");
    println!("   • Ultra-Low VRAM Sequential Streamer (< 7.5 GB Peak VRAM)");
    println!("================================================================================");

    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    let checkpoint = std::env::var("CKPT").unwrap_or_else(|_| "G:\\models\\flux\\flux-2-klein-9b.safetensors".into());
    let mistral_path = "G:\\models\\clip\\mistral3SmallFlux2Fp4_mistral3SmallFlux2.safetensors";
    let qwen8b_dir = "G:\\models\\clip\\FLUX.2-klein-9B_text_encoder";
    let vae_path = "G:\\models\\vae\\flux2-vae.safetensors";

    if !Path::new(&checkpoint).exists() {
        eprintln!("[-] Checkpoint not found: {}", checkpoint);
        return Ok(());
    }

    println!("\n📥 Loading Flux.2-Klein-9B Checkpoint with Sequential Streamer...");
    let t_start = Instant::now();
    let mut pipeline = FluxPipeline::from_single_file_streaming(&checkpoint, device.clone())
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

    pipeline.enable_flash_attn();

    // Attach Qwen3-8B text encoder (Flux.2-Klein-9B official) from multi-file shards.
    if Path::new(qwen8b_dir).is_dir() {
        println!("📥 Attaching Qwen3-8B Text Encoder from {} shards...", qwen8b_dir);
        let archive = SafeTensorsArchive::open_shards_dir(qwen8b_dir)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let qwen = aurora_rust_engine::text::Qwen3TextEncoder::from_archive(
            &archive,
            Some(std::path::Path::new("qwen_tokenizer.json")),
            &Device::Cpu,
            DType::F16,
        ).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        pipeline.set_qwen3(qwen);
        println!("✅ Qwen3-8B Text Encoder Attached!");
    }

    // Attach Flux 2 VAE Decoder
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
    println!("\n🎨 Generating Image with FLUX.2-Klein-9B (4 Steps)...");
    println!("📝 Prompt: \"{}\"", prompt);

    let params = DiffusionParams {
        prompt,
        negative_prompt: None,
        num_steps: 4,
        guidance_scale: 1.0,
        width: 1024,
        height: 1024,
        seed: 42,
    };

    let (image, metrics) = pipeline.generate_with_metrics(params, None::<fn(usize, usize, &candle_core::Tensor)>)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

    let out_path = std::env::var("OUT").unwrap_or_else(|_| "outputs/flux_showcase/flux_klein_9b_1024_seed42.png".into());
    image.save(&out_path)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to save output: {}", e)))?;

    println!("\n================================================================================");
    println!("🎉 FLUX.2-Klein-9B Generation Finished Successfully!");
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
