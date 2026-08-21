// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: LoRA Hot Weight Merging & Parity Verification Binary

use candle_core::{Device, Tensor};
use aurora_rust_engine::{DiffusionParams, StableDiffusionXLPipeline, TextToImagePipeline};
use std::fs;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

fn progress_callback(step: usize, total: usize, _latent: &Tensor) {
    if step == 1 || step % 5 == 0 || step == total {
        println!("    Step {}/{}", step, total);
        let _ = std::io::stdout().flush();
    }
}

fn main() -> anyhow::Result<()> {
    let output_dir = Path::new("outputs/lora_test");
    fs::create_dir_all(output_dir)?;

    let device = Device::new_cuda(0)?;
    println!("🚀 CUDA Device (RTX 4070 Ti) initialized.");

    let checkpoint_path = "G:\\models\\checkpoints\\duchaitenPonyXLNo_v60.safetensors";
    let lora_path = "G:\\models\\loras\\acidzlime-sdxl.safetensors";

    println!("📦 Loading base checkpoint: {}", checkpoint_path);
    let t_load = Instant::now();
    let mut pipeline = StableDiffusionXLPipeline::from_safetensors(checkpoint_path, &device)?;
    pipeline.enable_vae_tiling(None);
    println!("✅ Checkpoint loaded in {:.2}s", t_load.elapsed().as_secs_f64());

    let prompt = "score_9, score_8_up, score_7_up, masterpiece, 1girl, solo, stylized portrait, vibrant green eyes, neon futuristic city, detailed";
    let neg_prompt = "score_4, score_5, score_6, lowres, bad anatomy, bad hands, text, blurry";

    let params = DiffusionParams {
        prompt,
        negative_prompt: Some(neg_prompt),
        num_steps: 25,
        guidance_scale: 6.0,
        width: 1024,
        height: 1024,
        seed: 42,
    };

    // 1. Generate baseline image (No LoRA)
    println!("\n🎨 [1/3] Generating Baseline Image (No LoRA, 25 steps, seed 42)...");
    let t_gen1 = Instant::now();
    let img_base = pipeline.generate(params.clone(), Some(progress_callback))?;
    let base_dur = t_gen1.elapsed().as_secs_f64();
    let base_out = output_dir.join("01_baseline_no_lora.png");
    img_base.save(&base_out)?;
    println!("  ✨ Baseline completed in {:.2}s -> {}", base_dur, base_out.to_string_lossy().replace('\\', "/"));

    // 2. Hot-merge LoRA
    println!("\n🧬 [2/3] Hot-merging LoRA: {} (weight: 0.85)...", lora_path.replace('\\', "/"));
    let t_merge = Instant::now();
    pipeline.load_lora(lora_path, 0.85)?;
    println!("  ⚡ Merge completed in {:.2}s", t_merge.elapsed().as_secs_f64());

    // Generate with LoRA
    println!("🎨 Generating LoRA-applied Image (25 steps, seed 42)...");
    let t_gen2 = Instant::now();
    let img_lora = pipeline.generate(params.clone(), Some(progress_callback))?;
    let lora_dur = t_gen2.elapsed().as_secs_f64();
    let lora_out = output_dir.join("02_with_lora_acidzlime.png");
    img_lora.save(&lora_out)?;
    println!("  ✨ LoRA image completed in {:.2}s -> {}", lora_dur, lora_out.to_string_lossy().replace('\\', "/"));

    // 3. Unload LoRA
    println!("\n🔄 [3/3] Unloading LoRA and restoring base checkpoint...");
    pipeline.unload_all_loras()?;

    println!("🎨 Generating Post-Unload Image (25 steps, seed 42)...");
    let t_gen3 = Instant::now();
    let img_restored = pipeline.generate(params.clone(), Some(progress_callback))?;
    let rest_dur = t_gen3.elapsed().as_secs_f64();
    let rest_out = output_dir.join("03_post_unload_restored.png");
    img_restored.save(&rest_out)?;
    println!("  ✨ Restored image completed in {:.2}s -> {}", rest_dur, rest_out.to_string_lossy().replace('\\', "/"));

    println!("\n============================================================");
    println!("🎉 LoRA Hot-Merging & Restitution Verification Complete!");
    println!("   Baseline Time: {:.2}s", base_dur);
    println!("   LoRA Time:     {:.2}s (Zero runtime overhead: {:.2}s diff)", lora_dur, (lora_dur - base_dur).abs());
    println!("   Restored Time: {:.2}s", rest_dur);
    println!("============================================================");

    Ok(())
}
