use candle_core::quantized::{gguf_file, QMatMul};
use candle_core::{Device, Module, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dev = aurora_rust_engine::device::auto_device()?;

    println!("Testing QMatMul on device: {:?}", dev);
    let path = "/models/llm/Phi-4-reasoning-plus-Q4_K_M.gguf";
    if !std::path::Path::new(path).exists() {
        println!("Model file {} not present on this machine, skipping local test", path);
        return Ok(());
    }

    let file = std::fs::File::open(path)?;
    let mut reader = std::io::BufReader::new(file);
    let content = gguf_file::Content::read(&mut reader)?;
    println!("GGUF header loaded. Reading blk.0.attn_qkv.weight onto device...");
    let qtensor = content.tensor(&mut reader, "blk.0.attn_qkv.weight", &dev)?;
    println!("QTensor loaded: dtype {:?}, shape {:?}", qtensor.dtype(), qtensor.shape());
    let qmatmul = QMatMul::from_qtensor(qtensor)?;
    println!("QMatMul created successfully!");

    let input = Tensor::randn(0.0f32, 1.0f32, (1, 1, 5120), &dev)?;
    let out = qmatmul.forward(&input)?;
    println!("QMatMul forward output: shape {:?}, dtype {:?}", out.shape(), out.dtype());
    Ok(())
}
