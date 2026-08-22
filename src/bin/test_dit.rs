// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: Verification unit test for Flux.1 and SD 3.5 MMDiT blocks

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use aurora_rust_engine::diffusion::dit::{FluxConfig, FluxTransformer};

fn main() -> anyhow::Result<()> {
    println!("================================================================================");
    println!("🧪 Testing Pure Rust MMDiT (Flux.1 / SD 3.5) Architecture & Forward Pass...");
    println!("================================================================================\n");

    let device = Device::Cpu;
    let dtype = DType::F32;

    // Create a compact miniature MMDiT config for fast structural validation
    let mini_config = FluxConfig {
        in_channels: 64,
        out_channels: 64,
        hidden_size: 256,
        num_heads: 4,
        num_double_blocks: 2,
        num_single_blocks: 2,
        mlp_ratio: 2,
        theta: 10_000.0,
        guidance_embed: true,
    };

    let vb = VarBuilder::zeros(dtype, &device);
    let transformer = FluxTransformer::new(mini_config, vb)?;

    // Dummy Image tokens: Batch=1, 64 patches (equiv to 16x16 latent grid), 64 channels
    let img_tokens = Tensor::zeros((1, 64, 64), dtype, &device)?;
    // Dummy Text tokens: Batch=1, 77 sequence tokens, 4096 dim (T5-XXL)
    let txt_tokens = Tensor::zeros((1, 77, 4096), dtype, &device)?;
    // Timestep and guidance
    let timestep = Tensor::from_slice(&[500.0f32], (1,), &device)?;
    let guidance = Tensor::from_slice(&[3.5f32], (1,), &device)?;

    println!("⚡ Executing Multimodal DoubleStream + SingleStream Transformer Forward Pass...");
    let out = transformer.forward(&img_tokens, &txt_tokens, &timestep, Some(&guidance))?;

    println!("✅ MMDiT Output Shape: {:?}", out.shape().dims());
    assert_eq!(out.shape().dims(), &[1, 64, 64]);

    println!("\n🎉 Pure Rust MMDiT (Flux.1 / SD 3.5) Architecture Validated Successfully!");
    Ok(())
}
