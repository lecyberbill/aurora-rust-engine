use candle_core::{Device, quantized::gguf_file};
use std::fs::File;

fn main() -> anyhow::Result<()> {
    println!("Connecting to ROCm device 0...");
    #[cfg(feature = "rocm")]
    let dev = Device::new_rocm(0)?;
    #[cfg(not(feature = "rocm"))]
    let dev = Device::Cpu;

    println!("Opening GGUF...");
    let mut file = File::open("/models/llm/gemma-4-31B-it-UD-Q4_K_XL.gguf")?;
    let content = gguf_file::Content::read(&mut file)?;
    println!("Tensors in content: {}", content.tensor_infos.len());

    println!("Now calling ModelWeights::from_gguf...");
    let model = aurora_rust_engine::models::quantized_gemma3::ModelWeights::from_gguf(content, &mut file, &dev)?;
    println!("SUCCESSFULLY CREATED ModelWeights! Layers: {}", 60);
    Ok(())
}
