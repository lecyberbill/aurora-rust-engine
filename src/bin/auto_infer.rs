// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: Demonstrate the `transformers`-like AutoModel facade (a few lines to infer)

use candle_core::{Device, DType};
use aurora_rust_engine::hub::ModelHub;
use aurora_rust_engine::models::{downcast_model, AutoModel, DiffusionModel, ImageGenerationModel, ModelLoadConfig};
use aurora_rust_engine::traits::DiffusionParams;

// Shows the shape of the public API without tying it to a real (possibly slow) checkpoint.
// Swap the repo ids / device for real ones: the whole point is that loading + inferring ANY model
// takes only these few lines instead of wiring pipelines by hand.
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    let dtype = if device.is_cuda() { DType::F16 } else { DType::F32 };

    let hub = ModelHub::from_env()?;

    // 1) Load a diffusion model (Flux.2-Dev) from a HF repo or local path — one line.
    let cfg = ModelLoadConfig::new("local-or-repo/flux2-dev", device.clone(), dtype);
    let model = AutoModel::from_pretrained(&hub, cfg)?;

    // 2) Downcast the boxed `AnyModel` to the concrete model, then drive it as a generator.
    let mut inner = model.lock().unwrap();
    let gen = downcast_model::<DiffusionModel>(&mut *inner)
        .ok_or_else(|| anyhow::anyhow!("model is not a diffusion model"))?;

    // 3) Generate in ~3 lines.
    let rgb = ImageGenerationModel::generate_t2i(
        gen,
        DiffusionParams {
            prompt: "a fox in a snowy forest",
            num_steps: 8,
            guidance_scale: 3.5,
            width: 512,
            height: 512,
            seed: 42,
            negative_prompt: None,
        },
        None,
    )?;
    rgb.save("outputs/auto_model_fox.png")?;

    println!("✅ saved outputs/auto_model_fox.png (kind: {:?})", inner.kind());
    Ok(())
}
