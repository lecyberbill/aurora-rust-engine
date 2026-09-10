// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: SD3.5 Large end-to-end T2I test (transformer + CLIP-L/G + T5-XXL + 16ch VAE)

use std::collections::HashMap;
use std::path::Path;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use aurora_rust_engine::diffusion::vae_flux::FluxVaeDecoder;
use aurora_rust_engine::pipelines::FluxPipeline;
use aurora_rust_engine::text::{ClipTextEncoder, OpenClipTextEncoder, T5TextEncoder};
use aurora_rust_engine::traits::DiffusionParams;
use aurora_rust_engine::weights::SafeTensorsArchive;

/// Build a `VarBuilder` that exposes a single safetensors file's keys verbatim.
fn vb_from(path: &str, device: &Device, dtype: DType) -> Result<VarBuilder<'static>, Box<dyn std::error::Error>> {
    let a = SafeTensorsArchive::open(std::path::PathBuf::from(path))?;
    let mut map = HashMap::new();
    for k in a.tensor_names() {
        map.insert(k.clone(), a.get_tensor(&k, device, dtype)?);
    }
    Ok(VarBuilder::from_tensors(map, dtype, device))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    let dtype = if device.is_cuda() { DType::F16 } else { DType::F32 };

    let ckpt = std::env::var("CKPT").unwrap_or_else(|_| "G:\\models\\SD3\\stableDiffusion35Fp8_v35LargeTurbo.safetensors".into());
    let clip_l_path = "G:\\models\\clip\\clip_l.safetensors";
    let clip_g_path = "G:\\models\\clip\\clip_g.safetensors";
    let t5_path = "G:\\models\\clip\\t5xxl_fp16.safetensors";
    let vae_path = "G:\\models\\vae\\sd3_vae.safetensors";
    let prompt = std::env::var("PROMPT").unwrap_or_else(|_| "a majestic white wolf on a snowy cliff at golden hour, photorealistic, 8k".into());
    let steps: usize = std::env::var("STEPS").ok().and_then(|s| s.parse().ok()).unwrap_or(20);
    let guidance: f64 = std::env::var("GUIDANCE").ok().and_then(|s| s.parse().ok()).unwrap_or(4.5);
    let width: usize = std::env::var("WIDTH").ok().and_then(|s| s.parse().ok()).unwrap_or(512);
    let height: usize = std::env::var("HEIGHT").ok().and_then(|s| s.parse().ok()).unwrap_or(512);

    println!("📥 Loading SD3.5 transformer: {ckpt}");
    let mut pipe = FluxPipeline::from_single_file_streaming(&ckpt, device.clone())?;

    // Text encoders (CPU to keep the 9 GB T5 off the already-busy GPU).
    println!("📥 Building CLIP-L / CLIP-G / T5-XXL (CPU)...");
    let mut clip_l = ClipTextEncoder::new_sd15(vb_from(clip_l_path, &Device::Cpu, DType::F16)?)?;
    let _ = clip_l.load_tokenizer("clip_tokenizer.json");
    let mut clip_g = OpenClipTextEncoder::new_sdxl(vb_from(clip_g_path, &Device::Cpu, DType::F16)?)?;
    clip_g.load_tokenizer("openclip_tokenizer.json")?;
    // T5-XXL overflows to all-NaN in F16 -> load/run it in F32 (CPU).
    let t5 = T5TextEncoder::new(vb_from(t5_path, &Device::Cpu, DType::F32)?, Some(Path::new("t5xxl_tokenizer.json")))?;
    pipe.set_sd35_encoders(t5, clip_l);
    pipe.set_openclip_g(clip_g);

    // 16-channel SD3 VAE decoder.
    println!("📥 Building SD3 VAE decoder...");
    let vae_vb = vb_from(vae_path, &device, dtype)?;
    pipe.set_vae(FluxVaeDecoder::new(vae_vb)?);

    println!("🎨 Generating SD3.5 ({steps} steps, {width}x{height}, guidance {guidance})...");
    let (img, m) = pipe.generate_with_metrics(
        DiffusionParams {
            prompt: &prompt,
            negative_prompt: None,
            num_steps: steps,
            guidance_scale: guidance,
            width,
            height,
            seed: 42,
        },
        None::<fn(usize, usize, &Tensor)>,
    )?;

    let out = "outputs/sd35_test.png";
    img.save(out)?;
    println!("✅ saved {out} — UNet {:.1}s ({:.2} it/s), VAE {:.1}s, total {:.1}s",
        m.unet_total_ms / 1000.0, m.unet_it_per_sec, m.vae_decode_ms / 1000.0, m.total_wallclock_ms / 1000.0);
    Ok(())
}
