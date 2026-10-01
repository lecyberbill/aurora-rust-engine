// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: Pure Rust Vision-Language Model CLI for Visual Question Answering & Captioning

use std::path::PathBuf;
use std::time::Instant;
use candle_core::Device;
use image::ImageReader;
use aurora_rust_engine::{select_device, VlmParams, VlmPipeline};

fn main() -> anyhow::Result<()> {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║       AURORA RUST ENGINE — VISION-LANGUAGE MODEL (VLM)     ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    let args: Vec<String> = std::env::args().collect();
    let model_dir = args.get(1).map(PathBuf::from).unwrap_or_else(|| {
        let candidates = [
            "D:\\models\\vlm\\Qwen2.5-VL-3B",
            "D:\\models\\vlm\\PaliGemma-2-3B",
            "G:\\models\\LLM\\Qwen2-VL-7B",
            "models/vlm",
        ];
        for c in candidates {
            if std::path::Path::new(c).exists() {
                return PathBuf::from(c);
            }
        }
        PathBuf::from("models/vlm")
    });

    let image_path = args.get(2).map(PathBuf::from);
    let prompt = args.get(3).cloned().unwrap_or_else(|| "Describe this image in detail.".to_string());

    println!("[VLM] Model dir:  {}", model_dir.display());
    if let Some(ref img_p) = image_path {
        println!("[VLM] Input image: {}", img_p.display());
    } else {
        println!("[VLM] Mode: Pure Text (No Image)");
    }
    println!("[VLM] Prompt:      \"{}\"", prompt);

    if !model_dir.exists() {
        eprintln!("\n[ERROR] Model directory '{}' not found.", model_dir.display());
        eprintln!("Usage: cargo run --bin test_vlm -- <model_dir> [image.png] [prompt]");
        std::process::exit(1);
    }

    let device = select_device(true).unwrap_or(Device::Cpu);
    println!("[VLM] Device: {:?}", device);

    println!("[VLM] Loading weights & tokenizer...");
    let t0 = Instant::now();
    let mut pipeline = VlmPipeline::from_pretrained(&model_dir, &device)?;
    println!("[VLM] Model loaded in {:.2}s", t0.elapsed().as_secs_f32());

    let image = if let Some(ref p) = image_path {
        if p.exists() {
            println!("[VLM] Opening image: {}...", p.display());
            Some(ImageReader::open(p)?.decode()?)
        } else {
            eprintln!("[WARN] Image '{}' not found, proceeding in text-only mode.", p.display());
            None
        }
    } else {
        None
    };

    let params = VlmParams {
        max_tokens: 256,
        temperature: 0.7,
        top_p: 0.9,
        top_k: 40,
        repetition_penalty: 1.1,
        ..Default::default()
    };

    println!("[VLM] Generating response (streaming tokens)...");
    let t_gen = Instant::now();
    let response = pipeline.generate(image.as_ref(), &prompt, &params)?;
    let gen_dur = t_gen.elapsed();

    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║ VLM RESPONSE                                               ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!("{}\n", response);
    println!("⏱️ Generation time: {:.2}s", gen_dur.as_secs_f32());

    Ok(())
}
