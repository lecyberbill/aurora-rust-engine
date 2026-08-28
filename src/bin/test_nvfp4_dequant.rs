// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: Verification of Pure Rust NVFP4 Dequantization on Mistral Weights

use candle_core::{DType, Device, Result};
use std::path::PathBuf;
use std::time::Instant;
use aurora_rust_engine::text::mistral::dequantize_nvfp4;
use aurora_rust_engine::weights::SafeTensorsArchive;

fn main() -> Result<()> {
    println!("================================================================================");
    println!("🚀 AURORA PURE RUST NVFP4 DEQUANTIZER UNIT TEST");
    println!("   • Format: NVIDIA FP4 (E2M1) + FP8 Block Scales (16-wide) + FP32 Scale");
    println!("================================================================================");

    let path = "G:\\models\\clip\\mistral3SmallFlux2Fp4_mistral3SmallFlux2.safetensors";
    let archive = SafeTensorsArchive::open(PathBuf::from(path))
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

    let qx = archive.get_tensor("model.layers.0.self_attn.q_proj.weight", &Device::Cpu, DType::U8)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    let bs = archive.get_tensor("model.layers.0.self_attn.q_proj.weight_scale", &Device::Cpu, DType::F32)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    let s2 = archive.get_tensor("model.layers.0.self_attn.q_proj.weight_scale_2", &Device::Cpu, DType::F32)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    let per_tensor_scale = s2.to_vec0::<f32>()?;

    println!("📥 Input Tensors:");
    println!("   • Packed QX: {:?}", qx.shape());
    println!("   • Block Scales: {:?}", bs.shape());
    println!("   • Per-Tensor Scale: {}", per_tensor_scale);

    let t0 = Instant::now();
    let dequantized = dequantize_nvfp4(&qx, &bs, per_tensor_scale, DType::F32, &Device::Cpu)?;
    let dt = t0.elapsed().as_secs_f64();

    println!("\n✅ Dequantization finished in {:.4}s ({:.2} MB/s)", dt, (4096.0 * 5120.0 * 4.0 / 1024.0 / 1024.0) / dt);
    println!("📊 Result Tensor:");
    println!("   • Shape: {:?}", dequantized.shape());
    println!("   • Mean:  {:.8}", dequantized.mean_all()?.to_vec0::<f32>()?);
    println!("   • Min:   {:.8}", dequantized.min_all()?.to_vec0::<f32>()?);
    println!("   • Max:   {:.8}", dequantized.max_all()?.to_vec0::<f32>()?);

    Ok(())
}
