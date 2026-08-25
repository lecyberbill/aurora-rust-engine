// [WFGY] Zone: SAFE | λ: 0.15 | Fallbacks: 0 | Action: Diagnose Flux VAE and latents output values

use candle_core::Device;
use aurora_rust_engine::traits::DiffusionParams;
use aurora_rust_engine::pipelines::FluxPipeline;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    println!("🔍 Diagnosing Flux Latents and VAE output values...");

    let device = Device::new_cuda(0)?;
    let checkpoint = "G:\\models\\flux\\flux1SchnellFp8_schnellFp8.safetensors";

    if !Path::new(checkpoint).exists() {
        println!("[-] Checkpoint not found: {}", checkpoint);
        return Ok(());
    }

    let mut pipeline = FluxPipeline::from_single_file(checkpoint, device.clone())?;

    let params = DiffusionParams {
        prompt: "test",
        negative_prompt: None,
        num_steps: 1, // 1 step to quickly inspect values
        guidance_scale: 1.0,
        width: 512,
        height: 512,
        seed: 42,
    };

    let (image, _metrics) = pipeline.generate_with_metrics(params, None::<fn(usize, usize, &candle_core::Tensor)>)?;
    let (w, h) = image.dimensions();
    println!("📊 Generated Image Dimensions: {}x{}", w, h);

    let raw_bytes = image.as_raw();
    let min_val = raw_bytes.iter().min().copied().unwrap_or(0);
    let max_val = raw_bytes.iter().max().copied().unwrap_or(0);
    let mean_val = raw_bytes.iter().map(|&x| x as f64).sum::<f64>() / raw_bytes.len() as f64;

    println!("🎨 Pixel Stats: Min = {}, Max = {}, Mean = {:.2}", min_val, max_val, mean_val);
    println!("💾 Total Image Bytes: {}", raw_bytes.len());

    Ok(())
}
