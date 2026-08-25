// [WFGY] Zone: SAFE | λ: 0.15 | Fallbacks: 0 | Action: Debug Block 1 weights for NaNs

use candle_core::{DType, Device};
use aurora_rust_engine::weights::SafeTensorsArchive;

fn main() -> anyhow::Result<()> {
    println!("🔍 Inspecting raw weights of DoubleStreamBlock 1...");

    let device = Device::new_cuda(0)?;
    let checkpoint = "G:\\models\\flux\\flux1SchnellFp8_schnellFp8.safetensors";
    let archive = SafeTensorsArchive::open(checkpoint)?;

    for key in archive.keys() {
        if key.contains("double_blocks.1.") {
            let t = archive.get_tensor(key, &device, DType::F16)?;
            let v = t.flatten_all()?.to_dtype(DType::F32)?.to_device(&Device::Cpu)?.to_vec1::<f32>()?;
            let has_nan = v.iter().any(|x| x.is_nan());
            let min = v.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            println!("  Weight: {:<50} | Shape: {:?} | Min: {:.4}, Max: {:.4}, Has NaN: {}", key, t.dims(), min, max, has_nan);
        }
    }

    Ok(())
}
