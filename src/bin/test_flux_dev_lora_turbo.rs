// [WFGY] Zone: SAFE | λ: 0.25 | Fallbacks: 0 | Action: Verify FLUX.2-Dev + FLUX.2-Turbo LoRA (Diffusers transformer.*)

use candle_core::{DType, Device, Result};
use std::path::{Path, PathBuf};
use std::time::Instant;
use aurora_rust_engine::pipelines::FluxPipeline;
use aurora_rust_engine::traits::DiffusionParams;
use aurora_rust_engine::weights::{SafeTensorsArchive, WeightRouter};
use aurora_rust_engine::diffusion::vae_flux::FluxVaeDecoder;

fn main() -> Result<()> {
    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    let checkpoint = std::env::var("CKPT").unwrap_or_else(|_| "G:\\models\\flux\\flux2DevFp8Scaled_fp8Scaled.safetensors".into());
    let mistral_path = std::env::var("MISTRAL").unwrap_or_else(|_| "G:\\models\\clip\\FLUX.2-dev_text_encoder".into());
    let vae_path = std::env::var("VAE").unwrap_or_else(|_| "G:\\models\\vae\\flux2-vae.safetensors".into());
    let lora_path = std::env::var("LORA").unwrap_or_else(|_| "G:\\models\\loras\\flux2tatooR32.safetensors".into());
    let multiplier: f64 = std::env::var("MULT").ok().and_then(|s| s.parse().ok()).unwrap_or(0.85);
    let prompt = std::env::var("PROMPT").unwrap_or_else(|_| "a muscular man with intricate dragon tattoo on his chest and arm, studio portrait, photorealistic, 8k".into());
    let steps: usize = std::env::var("STEPS").ok().and_then(|s| s.parse().ok()).unwrap_or(8);
    let guidance: f64 = std::env::var("GUIDANCE").ok().and_then(|s| s.parse().ok()).unwrap_or(3.5);
    let width: usize = std::env::var("WIDTH").ok().and_then(|s| s.parse().ok()).unwrap_or(512);
    let height: usize = std::env::var("HEIGHT").ok().and_then(|s| s.parse().ok()).unwrap_or(512);

    if !Path::new(&checkpoint).exists() {
        eprintln!("[-] Checkpoint not found: {}", checkpoint);
        return Ok(());
    }
    if !Path::new(&lora_path).exists() {
        eprintln!("[-] LoRA not found: {}", lora_path);
        return Ok(());
    }

    let t_start = Instant::now();
    let mut pipeline = FluxPipeline::from_single_file_streaming(&checkpoint, device.clone())
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    pipeline.enable_flash_attn();

    if Path::new(&mistral_path).exists() {
        let mistral_dev = if device.is_cuda() { device.clone() } else { Device::Cpu };
        println!("📥 Attaching Mistral-3 Prompt Encoder ({} steps -> 40-layer VLM)...", steps);
        let mistral = if Path::new(&mistral_path).is_dir() {
            aurora_rust_engine::text::Mistral3TextEncoder::from_dir(
                &mistral_path,
                Some(std::path::Path::new("mistral_tokenizer.json")),
                mistral_dev,
                DType::F16,
            )
        } else {
            aurora_rust_engine::text::Mistral3TextEncoder::from_safetensors(
                &mistral_path,
                Some(std::path::Path::new("mistral_tokenizer.json")),
                mistral_dev,
                DType::F16,
            )
        }.map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        pipeline.set_mistral(mistral);
    }

    if Path::new(&vae_path).exists() {
        let vae_archive = if Path::new(&vae_path).is_dir() {
            SafeTensorsArchive::open_shards_dir(&vae_path)
        } else {
            SafeTensorsArchive::open(PathBuf::from(&vae_path))
        }.map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let vae_router = WeightRouter::new(&vae_archive, device.clone(), DType::F16);
        let vae_vb = vae_router.vae_var_builder().map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let decoder = FluxVaeDecoder::new(vae_vb)?;
        pipeline.set_vae(decoder);
    }

    println!("✅ Pipeline init in {:.2}s", t_start.elapsed().as_secs_f64());

    let params = DiffusionParams {
        prompt: prompt.as_str(),
        negative_prompt: None,
        num_steps: steps,
        guidance_scale: guidance,
        width,
        height,
        seed: 42,
    };

    let t0 = Instant::now();
    let (img_base, _) = pipeline.generate_with_metrics(params.clone(), None::<fn(usize, usize, &candle_core::Tensor)>)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    img_base.save("outputs/flux_showcase/flux_dev_turbo_baseline.png")
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    println!("baseline done in {:.2}s", t0.elapsed().as_secs_f64());

    println!("Loading FLUX.2-Turbo LoRA (Diffusers transformer.*): {}", lora_path);
    pipeline.load_lora(&lora_path, multiplier).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    println!("LoRA loaded (mult {})", multiplier);

    let t1 = Instant::now();
    let (img_lora, _) = pipeline.generate_with_metrics(params, None::<fn(usize, usize, &candle_core::Tensor)>)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    img_lora.save("outputs/flux_showcase/flux_dev_turbo_applied.png")
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    println!("lora render done in {:.2}s", t1.elapsed().as_secs_f64());

    Ok(())
}
