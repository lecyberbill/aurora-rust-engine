// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: End-to-end SDXL inference via the AutoModel facade (proves the "few lines" promise)

use candle_core::{Device, DType};
use aurora_rust_engine::models::{downcast_model, AutoModel, DiffusionModel, ImageGenerationModel};
use aurora_rust_engine::traits::DiffusionParams;

// Proves the facade: one AutoModel::from_local call loads an SDXL checkpoint (detecting the family
// automatically) and a handful of lines generate an image. Light-weight knobs (Euler, tiled VAE,
// FP8 + CPU offload on CUDA) are applied in one call for a low-memory run.
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    let dtype = if device.is_cuda() { DType::F16 } else { DType::F32 };

    let ckpt = std::env::var("CKPT").unwrap_or_else(|_| "G:\\models\\checkpoints\\sd_xl_base_1.0.safetensors".into());
    let prompt = std::env::var("PROMPT").unwrap_or_else(|_| "a majestic white wolf standing on a snowy cliff at golden hour, dramatic sky, photorealistic, 8k".into());
    let steps: usize = std::env::var("STEPS").ok().and_then(|s| s.parse().ok()).unwrap_or(8);

    println!("⏳ Loading SDXL via AutoModel::from_local (family auto-detect)...");
    let model = AutoModel::from_local(&ckpt, device, dtype)?;

    let mut inner = model.lock().unwrap();
    let gen = downcast_model::<DiffusionModel>(&mut *inner)
        .ok_or_else(|| anyhow::anyhow!("not a diffusion model"))?;

    // Optional override: OFF (default) keeps the pipeline exactly as the stress test uses it, so we
    // can attribute any slowdown to the facade vs to an explicit override. Set OVERRIDE=1 to apply.
    let override_knobs = std::env::var("OVERRIDE").ok().map(|v| v == "1").unwrap_or(false);
    if override_knobs {
        println!("🧪 Applying sdxl_fast_mode() override...");
        gen.sdxl_fast_mode()?;
    } else {
        println!("🧪 No override (default pipeline config, matching the stress test).");
    }

    println!("🎨 Generating ({steps} steps)...");
    let img = ImageGenerationModel::generate_t2i(
        gen,
        DiffusionParams {
            prompt: &prompt,
            negative_prompt: None,
            num_steps: steps,
            guidance_scale: 7.0,
            width: 512,
            height: 512,
            seed: 42,
        },
        None,
    )?;

    let out = "outputs/sdxl_auto_model.png";
    img.save(out)?;
    println!("✅ saved {out} — model kind {:?}, family {}", inner.kind(), inner.family());
    Ok(())
}
