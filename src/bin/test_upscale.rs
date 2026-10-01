// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: Pure Rust Super-Resolution CLI for Real-ESRGAN / 4x-UltraSharp

use std::path::PathBuf;
use std::time::Instant;
use candle_core::Device;
use image::ImageReader;
use aurora_rust_engine::{select_device, UpscaleParams, UpscalePipeline};

fn main() -> anyhow::Result<()> {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║       AURORA RUST ENGINE — REAL-ESRGAN SUPERSCALER         ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    let args: Vec<String> = std::env::args().collect();
    let model_path = args.get(1).map(PathBuf::from).unwrap_or_else(|| {
        // Search default candidate locations
        let candidates = [
            "D:\\models\\upscaler\\RealESRGAN_x4plus.safetensors",
            "D:\\models\\upscaler\\4x-UltraSharp.safetensors",
            "D:\\models\\upscaler\\RealESRGAN_x4plus_anime_6B.safetensors",
            "G:\\models\\upscaler\\4x-UltraSharp.safetensors",
            "RealESRGAN_x4plus.safetensors",
        ];
        for c in candidates {
            if std::path::Path::new(c).exists() {
                return PathBuf::from(c);
            }
        }
        PathBuf::from("RealESRGAN_x4plus.safetensors")
    });

    let input_path = args.get(2).map(PathBuf::from).unwrap_or_else(|| {
        let candidates = [
            "demo_input.png",
            "input.png",
            "test.png",
        ];
        for c in candidates {
            if std::path::Path::new(c).exists() {
                return PathBuf::from(c);
            }
        }
        PathBuf::from("demo_input.png")
    });

    let output_path = args.get(3).map(PathBuf::from).unwrap_or_else(|| PathBuf::from("upscaled_output.png"));

    println!("[ESRGAN] Model:  {}", model_path.display());
    println!("[ESRGAN] Input:  {}", input_path.display());
    println!("[ESRGAN] Output: {}", output_path.display());

    if !model_path.exists() {
        eprintln!("\n[ERROR] Model file '{}' not found.", model_path.display());
        eprintln!("Usage: cargo run --bin test_upscale -- <model.safetensors> <input.png> <output.png>");
        std::process::exit(1);
    }

    if !input_path.exists() {
        eprintln!("\n[ERROR] Input image '{}' not found.", input_path.display());
        eprintln!("Usage: cargo run --bin test_upscale -- <model.safetensors> <input.png> <output.png>");
        std::process::exit(1);
    }

    let device = select_device(true).unwrap_or(Device::Cpu);
    println!("[ESRGAN] Device: {:?}", device);

    println!("[ESRGAN] Loading weights...");
    let t0 = Instant::now();
    let pipeline = UpscalePipeline::load_from_safetensors(&model_path, &device)?;
    println!("[ESRGAN] Loaded successfully (Scale: {}x) in {:.2}s", pipeline.scale, t0.elapsed().as_secs_f32());

    println!("[ESRGAN] Opening input image...");
    let img = ImageReader::open(&input_path)?.decode()?;
    println!("[ESRGAN] Input resolution: {}x{}", img.width(), img.height());

    let params = UpscaleParams {
        tile_size: 512,
        tile_pad: 32,
    };

    println!("[ESRGAN] Upscaling image (tiled: {}px, pad: {}px)...", params.tile_size, params.tile_pad);
    let t_inf = Instant::now();
    let upscaled = pipeline.upscale_image(&img, &params)?;
    let inf_dur = t_inf.elapsed();

    println!("[ESRGAN] Upscaled resolution: {}x{} in {:.2}s", upscaled.width(), upscaled.height(), inf_dur.as_secs_f32());

    println!("[ESRGAN] Saving output to {}...", output_path.display());
    upscaled.save(&output_path)?;
    println!("[ESRGAN] Done! Output written to {}", output_path.display());

    Ok(())
}
