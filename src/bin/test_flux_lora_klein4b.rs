// [WFGY] Zone: SAFE | λ: 0.25 | Fallbacks: 0 | Action: Verify LoRA (Kohya-style `diffusion_model.*`) on FLUX.2-Klein-4B

use candle_core::{DType, Device, Result};
use std::path::PathBuf;
use std::time::Instant;
use aurora_rust_engine::pipelines::FluxPipeline;
use aurora_rust_engine::traits::DiffusionParams;
use aurora_rust_engine::diffusion::vae_flux::{FluxVaeDecoder, FluxVaeEncoder};
use aurora_rust_engine::weights::{SafeTensorsArchive, WeightRouter};

fn main() -> Result<()> {
    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    let checkpoint = std::env::var("CKPT").unwrap_or_else(|_| "G:\\models\\flux\\fluxKlein4BPro_v10.safetensors".into());
    let vae_path = std::env::var("VAE").unwrap_or_else(|_| "G:\\models\\vae\\flux2-vae.safetensors".into());
    let qwen_path = std::env::var("QWEN").unwrap_or_else(|_| "G:\\models\\clip\\qwen_3_4b.safetensors".into());
    let lora_path = std::env::var("LORA").unwrap_or_else(|_| "G:\\models\\loras\\NSGIRL-FLUX-2-Klein-4B-LoRA-By-MM744.safetensors".into());
    let multiplier: f64 = std::env::var("MULT").ok().and_then(|s| s.parse().ok()).unwrap_or(0.8);
    let prompt = std::env::var("PROMPT").unwrap_or_else(|_| "a beautiful woman portrait, soft studio lighting, elegant, photorealistic, 8k".into());

    if !std::path::Path::new(&checkpoint).exists() {
        eprintln!("[-] Checkpoint not found: {}", checkpoint);
        return Ok(());
    }
    if !std::path::Path::new(&lora_path).exists() {
        eprintln!("[-] LoRA not found: {}", lora_path);
        return Ok(());
    }

    let mut pipeline = FluxPipeline::from_single_file_streaming(&checkpoint, device.clone())
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    pipeline.enable_flash_attn();

    let qwen_archive = SafeTensorsArchive::open(PathBuf::from(&qwen_path))
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    let qwen_vb = candle_nn::VarBuilder::from_tensors(
        qwen_archive.tensor_names().into_iter().filter_map(|k| {
            qwen_archive.get_tensor(&k, &Device::Cpu, DType::F16).ok().map(|t| (k, t))
        }).collect(),
        DType::F16,
        &Device::Cpu,
    );
    let qwen = aurora_rust_engine::text::Qwen3TextEncoder::new(qwen_vb, Some(std::path::Path::new("qwen_tokenizer.json")))
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    pipeline.set_qwen3(qwen);

    let vae_archive = SafeTensorsArchive::open(PathBuf::from(&vae_path))
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    let vae_router = WeightRouter::new(&vae_archive, device.clone(), DType::F16);
    let vae_vb = vae_router.vae_var_builder().map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    let decoder = FluxVaeDecoder::new(vae_vb.clone())?;
    let encoder = FluxVaeEncoder::new(vae_vb)?;
    pipeline.set_vae(decoder);
    pipeline.set_vae_encoder(encoder);

    let params = DiffusionParams {
        prompt: prompt.as_str(),
        negative_prompt: None,
        num_steps: 8,
        guidance_scale: 1.0,
        width: 512,
        height: 512,
        seed: 42,
    };

    let t0 = Instant::now();
    let (img_base, _) = pipeline.generate_with_metrics(params.clone(), None::<fn(usize, usize, &candle_core::Tensor)>)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    img_base.save("outputs/flux_showcase/flux_klein4b_lora_baseline.png")
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    println!("baseline done in {:.2}s", t0.elapsed().as_secs_f64());

    println!("Loading LoRA (Kohya diffusion_model.*): {} (mult {})", lora_path, multiplier);
    pipeline.load_lora(&lora_path, multiplier).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    println!("LoRA loaded");

    let t1 = Instant::now();
    let (img_lora, _) = pipeline.generate_with_metrics(params, None::<fn(usize, usize, &candle_core::Tensor)>)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    img_lora.save("outputs/flux_showcase/flux_klein4b_lora_applied.png")
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    println!("lora render done in {:.2}s", t1.elapsed().as_secs_f64());

    Ok(())
}
