// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: SD3.5 Large end-to-end via the AutoModel facade (ModelDescriptor::sd35)

use candle_core::{DType, Device};
use aurora_rust_engine::models::{downcast_model, AutoModel, DiffusionModel, ImageGenerationModel, ModelDescriptor};
use aurora_rust_engine::traits::DiffusionParams;

// Proves the facade handles SD3.5: one `ModelDescriptor::sd35` (checkpoint + CLIP-L/G + T5 + VAE) and a
// few lines to generate, with the three text encoders and the 16-ch VAE wired automatically.
fn main() -> anyhow::Result<()> {
    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    let dtype = if device.is_cuda() { DType::F16 } else { DType::F32 };

    let ckpt = std::env::var("CKPT").unwrap_or_else(|_| "G:\\models\\SD3\\sd3.5_large.safetensors".into());
    let clip_l = std::env::var("CLIP_L").unwrap_or_else(|_| "G:\\models\\clip\\clip_l.safetensors".into());
    let clip_g = std::env::var("CLIP_G").unwrap_or_else(|_| "G:\\models\\clip\\clip_g.safetensors".into());
    let t5 = std::env::var("T5").unwrap_or_else(|_| "G:\\models\\clip\\t5xxl_fp16.safetensors".into());
    let vae = std::env::var("VAE").unwrap_or_else(|_| "G:\\models\\vae\\sd3_vae.safetensors".into());
    let prompt = std::env::var("PROMPT").unwrap_or_else(|_| {
        "a majestic white wolf standing on a snowy cliff at golden hour, dramatic clouds, photorealistic, 8k".into()
    });
    let steps: usize = std::env::var("STEPS").ok().and_then(|s| s.parse().ok()).unwrap_or(28);
    let guidance: f64 = std::env::var("GUIDANCE").ok().and_then(|s| s.parse().ok()).unwrap_or(3.5);
    let width: usize = std::env::var("WIDTH").ok().and_then(|s| s.parse().ok()).unwrap_or(512);
    let height: usize = std::env::var("HEIGHT").ok().and_then(|s| s.parse().ok()).unwrap_or(512);

    println!("⏳ Loading SD3.5 via AutoModel::from_descriptor(ModelDescriptor::sd35)...");
    let desc = ModelDescriptor::sd35("sd35-large", &ckpt, &clip_l, &clip_g, &t5, &vae);
    let model = AutoModel::from_descriptor(&desc, device, dtype)?;

    let mut inner = model.lock().unwrap();
    let gen = downcast_model::<DiffusionModel>(&mut *inner)
        .ok_or_else(|| anyhow::anyhow!("not a diffusion model"))?;

    println!("🎨 Generating ({steps} steps, guidance {guidance}, {width}x{height})...");
    let img = ImageGenerationModel::generate_t2i(
        gen,
        DiffusionParams {
            prompt: &prompt,
            negative_prompt: None,
            num_steps: steps,
            guidance_scale: guidance,
            width,
            height,
            seed: 42,
        },
        None,
    )?;

    let out = "outputs/sd35_auto_model.png";
    img.save(out)?;
    println!("✅ saved {out} — model kind {:?}, family {}", inner.kind(), inner.family());
    Ok(())
}
