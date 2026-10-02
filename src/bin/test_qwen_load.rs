use std::path::Path;
use candle_core::{DType, Device};
use aurora_rust_engine::weights::SafeTensorsArchive;
use aurora_rust_engine::text::Qwen3TextEncoder;

fn main() -> anyhow::Result<()> {
    let path = Path::new("/models/comfyui/text_encoders/qwen3vl_4b_fp8_scaled.safetensors");
    println!("Testing Qwen3TextEncoder loading from {:?}...", path);
    let archive = SafeTensorsArchive::open(path)?;
    println!("Archive opened successfully, keys: {}", archive.tensor_names().len());
    
    let enc = Qwen3TextEncoder::from_archive(
        &archive,
        Some(Path::new("qwen_tokenizer.json")),
        &Device::Cpu,
        DType::F32,
    );
    match enc {
        Ok(e) => {
            println!("✅ Qwen3TextEncoder successfully loaded! Has tokenizer: {}", e.has_tokenizer());
            let prompt = "a majestic white wolf on a snowy cliff at golden hour, photorealistic, 8k";
            let ctx = e.encode_krea2_12_layers(prompt, 0)?;
            println!("✅ encode_krea2_12_layers output shape: {:?}", ctx.dims());
        }
        Err(e) => {
            println!("❌ Qwen3TextEncoder failed to load: {:?}", e);
        }
    }
    Ok(())
}
