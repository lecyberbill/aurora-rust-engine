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
        let k_lower = k.to_lowercase();
        if k_lower.contains("expert") || k_lower.contains("gating") || k_lower.contains("template") || k_lower.contains("cache") || k_lower.contains("shortconv") || k_lower.contains("eos") || k_lower.contains("bos") {
            println!("  {k} => {:?}", v);
        }
    }
    Ok(())
}
