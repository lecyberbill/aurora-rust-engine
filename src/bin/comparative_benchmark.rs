// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: Optimized Rust (FlashAttention-2) vs Python Diffusers Comparative Benchmark

use candle_core::Device;
use aurora_rust_engine::{DiffusionParams, StableDiffusionXLPipeline};
use std::fs;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

const STRIKING_PROMPT: &str = "masterpiece, best quality, ultra-detailed, 1girl, solo, cyberpunk samurai warrior, glowing cherry blossoms, rainy futuristic neo-tokyo street, volumetric lighting, intricate reflections, 8k resolution, cinematic masterpiece";
const NEGATIVE_PROMPT: &str = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry";

struct BenchmarkTarget {
    name: &'static str,
    path: &'static str,
}

const TARGETS: &[BenchmarkTarget] = &[
    BenchmarkTarget {
        name: "duchaitenPonyXLNo_v60",
        path: "G:\\models\\checkpoints\\duchaitenPonyXLNo_v60.safetensors",
    },
    BenchmarkTarget {
        name: "Juggernaut-XL_v9_RunDiffusionPhoto_v2",
        path: "G:\\models\\checkpoints\\Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors",
    },
    BenchmarkTarget {
        name: "aniverseXL_v30",
        path: "G:\\models\\checkpoints\\aniverseXL_v30.safetensors",
    },
];

fn progress_cb(step: usize, total: usize, _latents: &candle_core::Tensor) {
    if step == 1 || step % 5 == 0 || step == total {
        print!("{} ", step);
        let _ = std::io::stdout().flush();
    }
}

fn main() -> anyhow::Result<()> {
    let out_dir = Path::new("outputs/stress_test/rust_flash_attn");
    fs::create_dir_all(out_dir)?;

    let device = Device::new_cuda(0)?;
    println!("============================================================");
    println!("🚀 Comparative Benchmark: Pure Rust (FlashAttention-2) vs Python Reference");
    println!("   Hardware: NVIDIA RTX 4070 Ti 12GB | Steps: 30 | Resolution: 1024x1024");
    println!("============================================================\n");

    for target in TARGETS {
        println!("------------------------------------------------------------");
        println!("📦 Target: {}", target.name);
        let path = Path::new(target.path);
        if !path.exists() {
            println!("  [-] Checkpoint not found: {}, skipping", target.path);
            continue;
        }

        let t_load = Instant::now();
        let mut pipeline = StableDiffusionXLPipeline::from_single_file(target.path, device.clone())?;
        let load_sec = t_load.elapsed().as_secs_f64();
        println!("  ✅ Loaded in {:.2}s", load_sec);

        let params = DiffusionParams {
            prompt: STRIKING_PROMPT,
            negative_prompt: Some(NEGATIVE_PROMPT),
            num_steps: 30,
            guidance_scale: 6.0,
            width: 1024,
            height: 1024,
            seed: 42,
        };

        print!("  🎨 Denoising (30 steps): ");
        let _ = std::io::stdout().flush();
        let (image, metrics) = pipeline.generate_with_metrics(params, Some(progress_cb))?;
        println!("done.");

        let out_filename = format!("{}_seed42.png", target.name);
        let out_path = out_dir.join(&out_filename);
        image.save(&out_path)?;

        println!("  📊 Telemetry Metrics:");
        println!("     • Text Encoders:  {:.2}s", metrics.prompt_encode_ms / 1000.0);
        println!("     • UNet Denoising: {:.2}s -> {:.2} it/s ({:.2} ms/step)", metrics.unet_total_ms / 1000.0, metrics.unet_it_per_sec, metrics.unet_step_avg_ms);
        println!("     • VAE Decode:     {:.2}s", metrics.vae_decode_ms / 1000.0);
        println!("     • Total Time:     {:.2}s", metrics.total_wallclock_ms / 1000.0);
        println!("     • Saved Image:    {}", out_path.to_string_lossy().replace('\\', "/"));
    }

    println!("\n============================================================");
    println!("🎉 Benchmark Completed Across All Target Models!");
    println!("============================================================");

    Ok(())
}
