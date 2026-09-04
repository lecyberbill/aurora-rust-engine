use candle_core::{Device, Result};
use std::path::Path;
use std::time::Instant;
use aurora_rust_engine::pipelines::FluxPipeline;
use aurora_rust_engine::traits::DiffusionParams;

fn main() -> Result<()> {
    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    let checkpoint = std::env::var("CKPT").unwrap_or_else(|_| "G:\\models\\flux\\flux1SchnellFp8_schnellFp8.safetensors".into());
    let lora_path = std::env::var("LORA").unwrap_or_else(|_| "G:\\models\\loras\\HoldPulledOffClothesFluxS1.1.0.safetensors".into());
    let multiplier: f64 = std::env::var("MULT").ok().and_then(|s| s.parse().ok()).unwrap_or(0.8);

    if !Path::new(&checkpoint).exists() {
        eprintln!("[-] Checkpoint not found: {}", checkpoint);
        return Ok(());
    }
    if !Path::new(&lora_path).exists() {
        eprintln!("[-] LoRA not found: {}", lora_path);
        return Ok(());
    }

    let mut pipeline = FluxPipeline::from_single_file_streaming(&checkpoint, device.clone())
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    pipeline.enable_flash_attn();

    let prompt = "a gorgeous portrait of a fox in a snowy forest, 8k";
    let params = DiffusionParams {
        prompt,
        negative_prompt: None,
        num_steps: 4,
        guidance_scale: 1.0,
        width: 512,
        height: 512,
        seed: 42,
    };

    // Baseline WITHOUT LoRA
    let t0 = Instant::now();
    let (img_base, _) = pipeline.generate_with_metrics(params.clone(), None::<fn(usize, usize, &candle_core::Tensor)>)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    img_base.save("outputs/flux_showcase/flux_lora_baseline.png")
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    println!("baseline done in {:.2}s", t0.elapsed().as_secs_f64());

    // Load LoRA
    println!("Loading LoRA: {} (mult {})", lora_path, multiplier);
    let t1 = Instant::now();
    pipeline.load_lora(&lora_path, multiplier)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    println!("LoRA loaded in {:.2}s", t1.elapsed().as_secs_f64());

    // Generate WITH LoRA
    let (img_lora, _) = pipeline.generate_with_metrics(params, None::<fn(usize, usize, &candle_core::Tensor)>)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    img_lora.save("outputs/flux_showcase/flux_lora_applied.png")
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    println!("lora render done");

    Ok(())
}
