// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: Comprehensive Multi-Prompt Multi-Aspect-Ratio SDXL Stress Test Benchmark

use candle_core::Device;
use aurora_rust_engine::{DiffusionParams, StableDiffusionXLPipeline};
use std::fs;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

struct PromptTest {
    id: &'static str,
    prompt: &'static str,
    negative_prompt: &'static str,
}

const PROMPTS: &[PromptTest] = &[
    PromptTest {
        id: "cyberpunk_warrior",
        prompt: "masterpiece, best quality, ultra-detailed, 1girl, solo, cyberpunk samurai warrior, glowing cherry blossoms, rainy futuristic neo-tokyo street, volumetric lighting, intricate reflections, 8k resolution, cinematic masterpiece",
        negative_prompt: "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, blurry",
    },
    PromptTest {
        id: "cinematic_portrait",
        prompt: "cinematic photo, 35mm photograph, a stylish detective in a wet trenchcoat standing under a street lamp in 1940s Chicago, heavy rain, moody atmosphere, film grain, photorealistic, sharp focus, 8k",
        negative_prompt: "drawing, painting, illustration, cartoon, 3d render, anime, deformed, disfigured, low quality, blurry",
    },
    PromptTest {
        id: "fantasy_landscape",
        prompt: "epic wide-angle landscape of floating islands with crystal waterfalls over a sea of glowing clouds, ancient stone temples, bioluminescent flora, golden hour sunlight, hyperrealistic, octane render",
        negative_prompt: "lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, signature, watermark, blurry",
    },
    PromptTest {
        id: "hyper_mech",
        prompt: "detailed futuristic mecha robot, complex machinery, hydraulic joints, carbon fiber plating, glowing energy core, battle damage, hangar background, ray tracing, unreal engine 5 render, highly detailed",
        negative_prompt: "bad proportions, low resolution, blurry, bad anatomy, simple background, flat colors",
    },
    PromptTest {
        id: "macro_wildlife",
        prompt: "award-winning macro photograph of a jewel-toned hummingbird hovering near an exotic orchid, translucent iridescent feathers, water droplets frozen in air, shallow depth of field, natural soft bokeh",
        negative_prompt: "blurry, low quality, bad focus, out of focus, distorted, dark, oversaturated, artificial",
    },
];

struct Resolution {
    name: &'static str,
    width: usize,
    height: usize,
}

const RESOLUTIONS: &[Resolution] = &[
    Resolution { name: "1024x1024_Square", width: 1024, height: 1024 },
    Resolution { name: "832x1216_Portrait", width: 832, height: 1216 },
    Resolution { name: "1216x832_Landscape", width: 1216, height: 832 },
];

fn progress_cb(step: usize, total: usize, _latents: &candle_core::Tensor) {
    if step == 1 || step % 10 == 0 || step == total {
        print!("{} ", step);
        let _ = std::io::stdout().flush();
    }
}

fn main() -> anyhow::Result<()> {
    let out_dir = Path::new("outputs/stress_test/matrix_5x3");
    fs::create_dir_all(out_dir)?;

    let device = Device::new_cuda(0)?;
    println!("============================================================");
    println!("🚀 Matrix Stress Test: 5 Prompts x 3 Aspect Ratios (15 Runs)");
    println!("   Target Checkpoint: Juggernaut-XL_v9_RunDiffusionPhoto_v2");
    println!("   Hardware: NVIDIA RTX 4070 Ti 12GB | Steps: 30");
    println!("============================================================\n");

    let checkpoint_path = "G:\\models\\checkpoints\\Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors";
    if !Path::new(checkpoint_path).exists() {
        eprintln!("[-] Checkpoint not found: {}", checkpoint_path);
        return Ok(());
    }

    let t_load = Instant::now();
    let mut pipeline = StableDiffusionXLPipeline::from_single_file(checkpoint_path, device)?;
    println!("✅ Pipeline loaded in {:.2}s\n", t_load.elapsed().as_secs_f64());

    let mut log_rows = Vec::new();

    let mut total_run_count = 0;
    let total_expected = PROMPTS.len() * RESOLUTIONS.len();

    for (p_idx, p_test) in PROMPTS.iter().enumerate() {
        for (_r_idx, res) in RESOLUTIONS.iter().enumerate() {
            total_run_count += 1;
            println!("------------------------------------------------------------");
            println!(
                "[{}/{}] Prompt #{}: '{}' | Res: {} ({}x{})",
                total_run_count, total_expected, p_idx + 1, p_test.id, res.name, res.width, res.height
            );

            let params = DiffusionParams {
                prompt: p_test.prompt,
                negative_prompt: Some(p_test.negative_prompt),
                num_steps: 30,
                guidance_scale: 6.5,
                width: res.width,
                height: res.height,
                seed: 42 + (total_run_count as u64),
            };

            print!("  🎨 Denoising (30 steps): ");
            let _ = std::io::stdout().flush();
            let (image, metrics) = pipeline.generate_with_metrics(params, Some(progress_cb))?;
            println!("done.");

            let filename = format!("{}_{}_{}x{}.png", p_test.id, res.name, res.width, res.height);
            let filepath = out_dir.join(&filename);
            image.save(&filepath)?;

            println!("  📊 Telemetry:");
            println!("     • Text Encoding:  {:.2}s", metrics.prompt_encode_ms / 1000.0);
            println!("     • UNet Denoising: {:.2}s -> {:.2} it/s ({:.2} ms/step)",
                metrics.unet_total_ms / 1000.0, metrics.unet_it_per_sec, metrics.unet_step_avg_ms);
            println!("     • VAE Decode:     {:.2}s", metrics.vae_decode_ms / 1000.0);
            println!("     • Total Time:     {:.2}s", metrics.total_wallclock_ms / 1000.0);
            println!("     • Image Saved:    {}", filepath.to_string_lossy().replace('\\', "/"));

            log_rows.push((
                p_test.id,
                res.name,
                res.width,
                res.height,
                metrics.prompt_encode_ms / 1000.0,
                metrics.unet_total_ms / 1000.0,
                metrics.unet_it_per_sec,
                metrics.vae_decode_ms / 1000.0,
                metrics.total_wallclock_ms / 1000.0,
                filename,
            ));
        }
    }

    println!("\n==================================================================================================================");
    println!("🏁 MATRIX BENCHMARK SUMMARY REPORT (15 RUNS)");
    println!("==================================================================================================================");
    println!("{:<20} | {:<18} | {:<7} | {:<7} | {:<7} | {:<7} | {:<7}",
        "Prompt", "Resolution", "Text", "UNet", "Speed", "VAE", "Total");
    println!("------------------------------------------------------------------------------------------------------------------");

    for row in &log_rows {
        println!("{:<20} | {:<18} | {:.2}s   | {:.2}s   | {:.2} it/s| {:.2}s   | {:.2}s",
            row.0, row.1, row.4, row.5, row.6, row.7, row.8);
    }
    println!("==================================================================================================================");

    Ok(())
}
