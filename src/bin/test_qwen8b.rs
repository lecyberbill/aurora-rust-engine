// Test loading Qwen3-8B from shards and encoding a prompt -> verify 12288 dim.
use candle_core::{DType, Device};
use aurora_rust_engine::text::Qwen3TextEncoder;
use aurora_rust_engine::weights::SafeTensorsArchive;

fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let dtype = DType::F16;

    let shards = [
        "G:\\models\\clip\\Qwen3-8B\\model-00001-of-00005.safetensors",
        "G:\\models\\clip\\Qwen3-8B\\model-00002-of-00005.safetensors",
        "G:\\models\\clip\\Qwen3-8B\\model-00003-of-00005.safetensors",
        "G:\\models\\clip\\Qwen3-8B\\model-00004-of-00005.safetensors",
        "G:\\models\\clip\\Qwen3-8B\\model-00005-of-00005.safetensors",
    ];
    println!("📥 Loading Qwen3-8B from {} shards...", shards.len());
    let archive = SafeTensorsArchive::open_shards(&shards)?;

    let mut enc = Qwen3TextEncoder::from_archive(&archive, Some(std::path::Path::new("qwen_tokenizer.json")), &device, dtype)?;
    println!("✅ Encoder built!");

    let out = enc.encode("a gorgeous portrait of an arctic fox with sapphire blue eyes", 512)?;
    println!("🏷️  encode dims = {:?}", out.dims());
    let f = out.to_dtype(DType::F32)?;
    let rms = f.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt();
    let amax = f.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
    println!("mean={:.5} rms={:.5} amax={:.3}", f.mean_all()?.to_scalar::<f32>()?, rms, amax);
    Ok(())
}
