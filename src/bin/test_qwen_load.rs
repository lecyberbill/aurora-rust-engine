use aurora_rust_engine::weights::SafeTensorsArchive;
use aurora_rust_engine::text::Qwen3TextEncoder;
use candle_core::{DType, Device};

fn main() -> anyhow::Result<()> {
    let path = "/models/comfyui/text_encoders/qwen3vl_4b_fp8_scaled.safetensors";
    println!("📂 Opening: {}", path);
    let archive = SafeTensorsArchive::open(path)?;
    println!("🔍 Attempting Qwen3TextEncoder::from_archive...");
    match Qwen3TextEncoder::from_archive(&archive, None, &Device::Cpu, DType::F32) {
        Ok(enc) => {
            println!("✅ Qwen3TextEncoder loaded successfully!");
            println!("   Tokenizer status: present={}", enc.has_tokenizer());
        }
        Err(e) => {
            println!("❌ Qwen3TextEncoder::from_archive failed: {:?}", e);
        }
    }
    Ok(())
}
