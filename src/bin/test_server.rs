// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: Server Route & Capabilities Unit Verification

use aurora_rust_engine::server::health_handler;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("🧪 Testing Aurora Rust Inference Server Handlers...");

    // Test health check endpoint handler directly
    let health = health_handler().await;
    println!("  ✅ Health check status: {}", health.status);
    println!("  ✅ Engine: {}", health.engine);
    println!("  ✅ CUDA Accelerated: {}", health.cuda_accelerated);
    println!("  ✅ FlashAttention-2: {}", health.flash_attention_2);
    println!("  ✅ LoRA In-Memory Merging: {}", health.lora_in_memory_merging);
    println!("  ✅ Multi-ControlNet Support: {}", health.controlnet_support);
    println!("  ✅ Inpainting Support: {}", health.inpainting_support);

    assert_eq!(health.status, "healthy");
    assert!(health.lora_in_memory_merging);
    assert!(health.controlnet_support);
    assert!(health.inpainting_support);

    println!("\n============================================================");
    println!("🎉 Aurora AI Microservice Route Handlers Validated!");
    println!("============================================================");

    Ok(())
}
