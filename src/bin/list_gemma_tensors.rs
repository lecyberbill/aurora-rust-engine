use std::fs::File;
use std::io::BufReader;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let file = File::open(&args[1])?;
    let mut reader = BufReader::new(file);
    let content = candle_core::quantized::gguf_file::Content::read(&mut reader)?;
    let mut keys: Vec<_> = content.tensor_infos.keys().cloned().collect();
    keys.sort();
    for k in keys {
        if k.starts_with("blk.5.") {
            let info = &content.tensor_infos[&k];
            println!("{k} => shape: {:?}, dtype: {:?}", info.shape.dims(), info.ggml_dtype);
        }
    }
    Ok(())
}
