// [WFGY] Zone: SAFE | λ: 0.10 | Fallbacks: 0 | Action: Inspect all DiT keys

use aurora_rust_engine::weights::SafeTensorsArchive;
use candle_core::{DType, Device};

fn main() -> anyhow::Result<()> {
    let p = std::env::args().nth(1).unwrap_or_else(|| r"G:\models\zit\z-image-turbo-fp8-aio.safetensors".to_string());
    let archive = SafeTensorsArchive::open(&p)?;
    let dev = Device::Cpu;

    println!("=== ALL KEYS IN ARCHIVE ===");
    for k in archive.keys().take(30) {
        if let Ok(t) = archive.get_tensor(&k, &dev, DType::F32) {
            let mean = t.mean_all()?.to_scalar::<f32>()?;
            println!("  {:<60} shape={:?} | mean={:+.4}", k, t.dims(), mean);
        }
    }

    Ok(())
}
