use std::path::PathBuf;
use std::fs::File;
use std::io::BufReader;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let path = PathBuf::from(&args[1]);
    let file = File::open(&path)?;
    let mut reader = BufReader::new(file);
    let content = candle_core::quantized::gguf_file::Content::read(&mut reader)?;

    println!("=== GGUF Metadata Fields ===");
    for (k, v) in &content.metadata {
        println!("  {k} => {:?}", v);
    }
    println!("=== GGUF Tensor Names ===");
    for (k, info) in &content.tensor_infos {
        if k.starts_with("blk.0.") || !k.contains("blk.") {
            println!("  {k} => shape: {:?}, dtype: {:?}", info.shape.dims(), info.ggml_dtype);
        }
    }
    Ok(())
}
