// [WFGY] Zone: SAFE | λ: 0.1 | Fallbacks: 0 | Action: Debug Klein step tensor norms and values

use candle_core::{DType, Device, Tensor};
use std::path::Path;
use aurora_rust_engine::traits::DiffusionParams;
use aurora_rust_engine::weights::{SafeTensorsArchive, WeightRouter};
use aurora_rust_engine::pipelines::flux::FluxPipeline;
use aurora_rust_engine::text::Qwen3TextEncoder;
use aurora_rust_engine::diffusion::vae_flux::FluxVaeDecoder;

fn main() -> anyhow::Result<()> {
    let device = Device::new_cuda(0)?;
    let dtype = DType::F16;

    let flux_path = "G:\\models\\flux\\fluxKlein4BPro_v10.safetensors";
    let qwen_path = "G:\\models\\clip\\qwen_3_4b.safetensors";
    let vae_path = "G:\\models\\vae\\flux2-vae.safetensors";

    println!("📥 Loading pipeline for Klein checkpoint...");
    let mut pipeline = FluxPipeline::from_single_file_streaming(flux_path, device.clone())?;

    println!("📥 Loading Qwen3...");
    let qwen_archive = SafeTensorsArchive::open(qwen_path)?;
    let mut qwen_map = std::collections::HashMap::new();
    for key in qwen_archive.keys() {
        if let Ok(t) = qwen_archive.get_tensor(key, &Device::Cpu, DType::F32) {
            qwen_map.insert(key.clone(), t);
        }
    }
    let qwen_vb = candle_nn::VarBuilder::from_tensors(qwen_map, DType::F32, &Device::Cpu);
    pipeline.set_qwen3(Qwen3TextEncoder::new(qwen_vb, Some(Path::new("qwen_tokenizer.json")))?);

    println!("📥 Loading VAE...");
    let vae_archive = SafeTensorsArchive::open(vae_path)?;
    let vae_router = WeightRouter::new(&vae_archive, device.clone(), DType::F16);
    let vae_vb = vae_router.vae_var_builder()?;
    pipeline.set_vae(FluxVaeDecoder::new(vae_vb)?);

    let params = DiffusionParams {
        prompt: "masterpiece, highly detailed, futuristic glowing robot, neon lights, 8k",
        negative_prompt: None,
        height: 1024,
        width: 1024,
        num_steps: 8,
        guidance_scale: 1.0,
        seed: 42,
    };

    println!("⚡ Executing generation with Step Diagnostics...");
    let (img, metrics) = pipeline.generate_with_metrics(params, Some(|step, total, latents: &Tensor| {
        let l_f32 = latents.to_dtype(DType::F32).unwrap();
        let mean = l_f32.mean_all().unwrap().to_vec0::<f32>().unwrap();
        let var = l_f32.sqr().unwrap().mean_all().unwrap().to_vec0::<f32>().unwrap();
        let std = var.sqrt();
        let min = l_f32.flatten_all().unwrap().to_vec1::<f32>().unwrap().into_iter().fold(f32::INFINITY, f32::min);
        let max = l_f32.flatten_all().unwrap().to_vec1::<f32>().unwrap().into_iter().fold(f32::NEG_INFINITY, f32::max);
        println!("   [Step {}/{}] mean: {:.4}, std: {:.4}, min: {:.4}, max: {:.4}", step, total, mean, std, min, max);
    }))?;

    img.save("outputs/flux_showcase/debug_klein_result.png")?;
    println!("✅ Done! Metrics: {:?}", metrics);
    Ok(())
}
