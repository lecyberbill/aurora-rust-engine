use std::path::PathBuf;
use std::time::Instant;
use candle_core::DType;
use aurora_rust_engine::device::auto_device;
use aurora_rust_engine::pipelines::ZImageTurboPipeline;
use aurora_rust_engine::traits::DiffusionParams;

fn main() -> anyhow::Result<()> {
    println!("================================================================================");
    println!("⚡ Aurora Engine: Z-Image Turbo Realtime DiT Validation (Milestone 17)");
    println!("================================================================================");

    let device = auto_device()?;
    let dtype = if device.is_cuda() { DType::BF16 } else { DType::F32 };
    println!("🎮 Target Device: {:?} | Precision: {:?}", device, dtype);

    let aio_path = PathBuf::from(r"G:\models\zit\z-image-turbo-fp8-aio.safetensors");
    let vae_path = PathBuf::from(r"G:\models\zit\zImageTurbo_vae.safetensors");

    if !aio_path.exists() {
        eprintln!("❌ Checkpoint not found: {:?}", aio_path);
        return Ok(());
    }

    let load_start = Instant::now();
    let mut pipeline = ZImageTurboPipeline::from_aio_checkpoint(
        &aio_path,
        if vae_path.exists() { Some(&vae_path) } else { None },
        &device,
        dtype,
    )?;
    println!("✅ Pipeline loaded in {:.2} s", load_start.elapsed().as_secs_f64());

    let prompt = "A cinematic futuristic sports car driving through a neon cyber city at night, 8k octane render, hyperdetailed";
    let params = DiffusionParams {
        prompt,
        negative_prompt: None,
        num_steps: 4, // 4-step Turbo inference
        guidance_scale: 1.0, // DiT Turbo is guidance-free / guidance distilled
        seed: 42,
        width: 512,
        height: 512,
    };

    println!("\n🎨 Running 4-step real-time generation...");
    let (image, metrics) = pipeline.generate(&params)?;

    let out_dir = PathBuf::from("output");
    std::fs::create_dir_all(&out_dir)?;
    let out_file = out_dir.join("zimage_turbo_test.png");
    image.save(&out_file)?;

    println!("================================================================================");
    println!("🎉 Generated image saved to: {:?}", out_file);
    println!("⏱️ Total Time: {:.2} ms", metrics.total_wallclock_ms);
    println!("   • Text Encoding: {:.2} ms", metrics.prompt_encode_ms);
    println!("   • Denoising (4 steps): {:.2} ms ({:.2} ms/step)", metrics.unet_total_ms, metrics.unet_step_avg_ms);
    println!("   • VAE Decode: {:.2} ms", metrics.vae_decode_ms);
    println!("================================================================================");

    Ok(())
}
