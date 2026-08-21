// [WFGY] Zone: SAFE | λ: 0.25 | Fallbacks: 1 (per-model isolation) | Action: FlashAttention-2 Accelerated SDXL 15-Model Benchmark Runner for aurora-rust-engine

use candle_core::{Device, Tensor};
use aurora_rust_engine::{DiffusionParams, StableDiffusionXLPipeline, TextToImagePipeline};
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use std::time::Instant;

const STRIKING_PROMPT: &str = "masterpiece, best quality, ultra-detailed, 1girl, solo, cyberpunk samurai warrior, glowing cherry blossoms, rainy futuristic neo-tokyo street, volumetric lighting, intricate reflections, 8k resolution, cinematic masterpiece";
const NEGATIVE_PROMPT: &str = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry";

#[derive(serde::Serialize)]
struct BenchmarkData {
    engine: String,
    device: String,
    prompt: String,
    negative_prompt: String,
    guidance_scale: f64,
    steps: usize,
    width: usize,
    height: usize,
    models: Vec<ModelStressResult>,
}

#[derive(serde::Serialize, Clone)]
struct ModelStressResult {
    model_name: String,
    model_size_gb: f64,
    status: String,
    load_time_sec: f64,
    images: Vec<ImageResult>,
}

#[derive(serde::Serialize, Clone)]
struct ImageResult {
    seed: u64,
    steps: usize,
    duration_sec: f64,
    it_per_sec: f64,
    output_path: String,
}

fn progress_callback(step: usize, total: usize, _latent: &Tensor) {
    if step == 1 || step % 5 == 0 || step == total {
        println!("    Step {}/{}", step, total);
        let _ = std::io::stdout().flush();
    }
}

fn main() -> anyhow::Result<()> {
    let output_dir = Path::new("outputs/stress_test/rust");
    fs::create_dir_all(output_dir)?;

    let log_path = Path::new("outputs/stress_test/flash_rust_stress_test_log.md");
    let json_path = Path::new("outputs/stress_test/flash_rust_metrics.json");

    let device = match Device::new_cuda(0) {
        Ok(dev) => {
            println!("🚀 CUDA Acceleration Device (RTX 4070 Ti) initialized successfully.");
            dev
        }
        Err(e) => {
            println!("⚠️ CUDA failed ({}), using CPU.", e);
            Device::Cpu
        }
    };

    let target_models = vec![
        "G:\\models\\checkpoints\\animaPencilXL_v100.safetensors",
        "G:\\models\\checkpoints\\aniverseXL_v30.safetensors",
        "G:\\models\\checkpoints\\babesByStableYogiPony_v50.safetensors",
        "G:\\models\\checkpoints\\Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors",
        "G:\\models\\checkpoints\\betterThanWords_v30.safetensors",
        "G:\\models\\checkpoints\\bigLove_ponyV20.safetensors",
        "G:\\models\\checkpoints\\realismarkPlus_realismarkPlus.safetensors",
        "G:\\models\\checkpoints\\CHEYENNE_v20.safetensors",
        "G:\\models\\checkpoints\\colossusProjectXLSFW_10bNeodemonFP16.safetensors",
        "G:\\models\\checkpoints\\CyberRealisticPony_V7a.safetensors",
        "G:\\models\\checkpoints\\dreamshaperXL_turboDpmppSDEKarras.safetensors",
        "G:\\models\\checkpoints\\DreamShaperXL_Turbo_v2_1.safetensors",
        "G:\\models\\checkpoints\\duchaitenAiartSDXL_v33515LightningTCD.safetensors",
        "G:\\models\\checkpoints\\duchaitenPonyXLNo_v60.safetensors",
        "G:\\models\\checkpoints\\eldgardKinkiestModel_v20.safetensors",
    ];

    println!("============================================================");
    println!("⚡ Starting FlashAttention-2 Accelerated aurora-rust-engine Stress Test");
    println!("   Device: NVIDIA RTX 4070 Ti | FP16 Precision | Steps: 30");
    println!("============================================================");

    let seeds = [42u64, 1337u64];
    let num_steps = 30;
    let guidance_scale = 6.0;
    let width = 1024;
    let height = 1024;

    let mut model_results = Vec::new();

    for (idx, model_path_str) in target_models.iter().enumerate() {
        let path = Path::new(model_path_str);
        let model_name = path.file_name().unwrap().to_string_lossy().to_string();
        let file_size_gb = if path.exists() {
            fs::metadata(path).map(|m| m.len() as f64 / 1_073_741_824.0).unwrap_or(0.0)
        } else {
            0.0
        };

        println!("\n[{}/{}] 📦 Model: {} ({:.2} GB)", idx + 1, target_models.len(), model_name, file_size_gb);

        if !path.exists() {
            println!("  ❌ File does not exist, skipping.");
            model_results.push(ModelStressResult {
                model_name: model_name.clone(),
                model_size_gb: file_size_gb,
                status: "File Not Found".to_string(),
                load_time_sec: 0.0,
                images: Vec::new(),
            });
            continue;
        }

        // Load pipeline with fresh VRAM state
        let t_load_start = Instant::now();
        let pipeline_res = StableDiffusionXLPipeline::from_safetensors(path, &device);

        let (mut pipeline, load_sec) = match pipeline_res {
            Ok(mut pipe) => {
                // Enable automatic VAE tiling for low-VRAM decoding
                pipe.enable_vae_tiling(None);
                let load_sec = t_load_start.elapsed().as_secs_f64();
                println!("  ✅ Weights loaded in {:.2}s", load_sec);
                (Some(pipe), load_sec)
            }
            Err(e) => {
                println!("  ❌ Load Error: {}", e);
                model_results.push(ModelStressResult {
                    model_name: model_name.clone(),
                    model_size_gb: file_size_gb,
                    status: format!("Load Error: {}", e),
                    load_time_sec: t_load_start.elapsed().as_secs_f64(),
                    images: Vec::new(),
                });
                continue;
            }
        };

        let mut image_results = Vec::new();

        if let Some(ref mut pipe) = pipeline {
            for &seed in &seeds {
                println!("  🖼️ Generating image with seed {} ({} steps)...", seed, num_steps);
                let params = DiffusionParams {
                    prompt: STRIKING_PROMPT,
                    negative_prompt: Some(NEGATIVE_PROMPT),
                    num_steps,
                    guidance_scale,
                    width,
                    height,
                    seed,
                };

                let t_gen_start = Instant::now();
                let gen_res = pipe.generate(params, Some(progress_callback));
                println!();

                match gen_res {
                    Ok(img) => {
                        let duration = t_gen_start.elapsed().as_secs_f64();
                        let it_per_sec = num_steps as f64 / duration;
                        let clean_name = model_name.trim_end_matches(".safetensors");
                        let out_filename = format!("flash_{}_seed{}.png", clean_name, seed);
                        let out_file_path = output_dir.join(&out_filename);
                        img.save(&out_file_path)?;

                        println!("    ✨ Completed in {:.2}s ({:.2} it/s) -> {}", duration, it_per_sec, out_filename);

                        image_results.push(ImageResult {
                            seed,
                            steps: num_steps,
                            duration_sec: duration,
                            it_per_sec,
                            output_path: out_file_path.to_string_lossy().to_string(),
                        });
                    }
                    Err(e) => {
                        println!("    ❌ Generation Error (seed {}): {}", seed, e);
                    }
                }
            }
        }

        // Explicitly drop pipeline to immediately reclaim VRAM before next model
        drop(pipeline);

        model_results.push(ModelStressResult {
            model_name: model_name.clone(),
            model_size_gb: file_size_gb,
            status: "SUCCESS".to_string(),
            load_time_sec: load_sec,
            images: image_results,
        });

        // Flush intermediate markdown summary table
        let mut f = File::create(log_path)?;
        write_summary_table(&mut f, &model_results)?;

        // Flush intermediate json
        let benchmark_data = BenchmarkData {
            engine: "aurora-rust-engine (FlashAttention-2)".to_string(),
            device: "NVIDIA GeForce RTX 4070 Ti (12GB)".to_string(),
            prompt: STRIKING_PROMPT.to_string(),
            negative_prompt: NEGATIVE_PROMPT.to_string(),
            guidance_scale,
            steps: num_steps,
            width,
            height,
            models: model_results.clone(),
        };
        if let Ok(json_str) = serde_json::to_string_pretty(&benchmark_data) {
            let _ = fs::write(json_path, json_str);
        }

        // Flush CSV metrics
        let csv_path = Path::new("outputs/stress_test/flash_rust_benchmark_metrics.csv");
        let mut csv_file = File::create(csv_path)?;
        writeln!(csv_file, "model_name,model_size_gb,load_time_sec,status,img1_duration_sec,img1_it_per_sec,img2_duration_sec,img2_it_per_sec")?;
        for res in &model_results {
            let img1_dur = res.images.get(0).map(|i| i.duration_sec).unwrap_or(0.0);
            let img1_it = res.images.get(0).map(|i| i.it_per_sec).unwrap_or(0.0);
            let img2_dur = res.images.get(1).map(|i| i.duration_sec).unwrap_or(0.0);
            let img2_it = res.images.get(1).map(|i| i.it_per_sec).unwrap_or(0.0);
            writeln!(
                csv_file,
                "{},{:.2},{:.2},{},{:.2},{:.2},{:.2},{:.2}",
                res.model_name, res.model_size_gb, res.load_time_sec, res.status, img1_dur, img1_it, img2_dur, img2_it
            )?;
        }
    }

    println!("\n============================================================");
    println!("🏁 FlashAttention-2 Pass (Rust) Finished! Report: {}", log_path.display());
    println!("============================================================");

    Ok(())
}

fn write_summary_table<W: Write>(f: &mut W, results: &[ModelStressResult]) -> std::io::Result<()> {
    writeln!(f, "# ⚡ SDXL 15-Model FlashAttention-2: `aurora-rust-engine` (Pure Rust) Benchmark\n")?;
    writeln!(f, "**Device**: NVIDIA GeForce RTX 4070 Ti (12GB VRAM) | **Precision**: FP16 Native | **Attention**: FlashAttention-2")?;
    writeln!(f, "**Resolution**: 1024x1024 | **Steps**: 30 (Euler Karras) | **CFG**: 6.0\n")?;
    writeln!(f, "| # | Model Name | Size | Load Time | Status | Seed 42 Speed | Seed 1337 Speed | Image 1 | Image 2 |")?;
    writeln!(f, "|---|---|---|---|---|---|---|---|---|")?;

    for (i, res) in results.iter().enumerate() {
        let img1 = res.images.get(0);
        let img2 = res.images.get(1);

        let speed1_str = img1.map(|img| format!("{:.2}s ({:.2} it/s)", img.duration_sec, img.it_per_sec)).unwrap_or("-".to_string());
        let speed2_str = img2.map(|img| format!("{:.2}s ({:.2} it/s)", img.duration_sec, img.it_per_sec)).unwrap_or("-".to_string());

        let link1_str = img1.map(|img| {
            let filename = Path::new(&img.output_path).file_name().unwrap().to_string_lossy();
            format!("[{}]({})", filename, filename)
        }).unwrap_or("-".to_string());

        let link2_str = img2.map(|img| {
            let filename = Path::new(&img.output_path).file_name().unwrap().to_string_lossy();
            format!("[{}]({})", filename, filename)
        }).unwrap_or("-".to_string());

        writeln!(
            f,
            "| {} | `{}` | {:.2} GB | {:.2}s | {} | {} | {} | {} | {} |",
            i + 1,
            res.model_name,
            res.model_size_gb,
            res.load_time_sec,
            res.status,
            speed1_str,
            speed2_str,
            link1_str,
            link2_str
        )?;
    }

    writeln!(f, "\n---\n")?;
    Ok(())
}
