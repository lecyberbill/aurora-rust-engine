// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: High-Performance Async AI Microservice Entrypoint

use candle_core::Device;
use aurora_rust_engine::{run_server, StableDiffusionXLPipeline};
use std::net::SocketAddr;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let device = Device::new_cuda(0)?;
    println!("🚀 CUDA Device (RTX 4070 Ti) initialized for Inference Microservice.");

    let checkpoint_path = "G:\\models\\checkpoints\\duchaitenPonyXLNo_v60.safetensors";
    println!("📦 Loading base SDXL model: {}", checkpoint_path.replace('\\', "/"));
    let pipeline = StableDiffusionXLPipeline::from_single_file(checkpoint_path, device)?;

    let addr: SocketAddr = "127.0.0.1:8080".parse()?;
    println!("🌟 Aurora Rust AI Inference Microservice Ready!");
    println!("   • Health check:  http://{}/api/v1/health", addr);
    println!("   • REST Generate: http://{}/api/v1/generate", addr);
    println!("   • WebSocket:     ws://{}/api/v1/ws", addr);

    run_server(pipeline, addr).await?;

    Ok(())
}
