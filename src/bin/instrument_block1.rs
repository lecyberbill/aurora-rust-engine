// [WFGY] Zone: SAFE | λ: 0.15 | Fallbacks: 0 | Action: Instrument DoubleStreamBlock forward step-by-step

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use aurora_rust_engine::weights::SafeTensorsArchive;
use aurora_rust_engine::diffusion::dit::blocks::DoubleStreamBlock;
use std::collections::HashMap;

fn main() -> anyhow::Result<()> {
    println!("🔍 Instrumenting Block 1 Step-by-Step...");

    let device = Device::new_cuda(0)?;
    let checkpoint = "G:\\models\\flux\\flux1SchnellFp8_schnellFp8.safetensors";
    let archive = SafeTensorsArchive::open(checkpoint)?;

    let mut tensors0 = HashMap::new();
    let mut tensors1 = HashMap::new();
    for key in archive.keys() {
        if let Some(s) = key.strip_prefix("model.diffusion_model.double_blocks.0.") {
            tensors0.insert(s.to_string(), archive.get_tensor(key, &device, DType::F16)?);
        }
        if let Some(s) = key.strip_prefix("model.diffusion_model.double_blocks.1.") {
            tensors1.insert(s.to_string(), archive.get_tensor(key, &device, DType::F16)?);
        }
    }

    let vb0 = VarBuilder::from_tensors(tensors0, DType::F16, &device);
    let vb1 = VarBuilder::from_tensors(tensors1, DType::F16, &device);
    let block0 = DoubleStreamBlock::new(3072, 24, 4, vb0)?;
    let block1 = DoubleStreamBlock::new(3072, 24, 4, vb1)?;

    let img0 = Tensor::randn(0f32, 1f32, (1, 1024, 3072), &device)?.to_dtype(DType::F16)?;
    let txt0 = Tensor::randn(0f32, 1f32, (1, 256, 3072), &device)?.to_dtype(DType::F16)?;
    let temb = Tensor::randn(0f32, 1f32, (1, 3072), &device)?.to_dtype(DType::F16)?;

    println!("⚡ Executing Block 0...");
    let (img1, txt1) = block0.forward(&img0, &txt0, &temb, None, None)?;
    let v1 = img1.flatten_all()?.to_dtype(DType::F32)?.to_device(&Device::Cpu)?.to_vec1::<f32>()?;
    println!("  Block 0 Out: Min = {:.4}, Max = {:.4}", v1.iter().cloned().fold(f32::INFINITY, f32::min), v1.iter().cloned().fold(f32::NEG_INFINITY, f32::max));

    println!("⚡ Executing Block 1...");
    let (img2, txt2) = block1.forward(&img1, &txt1, &temb, None, None)?;
    let v2 = img2.flatten_all()?.to_dtype(DType::F32)?.to_device(&Device::Cpu)?.to_vec1::<f32>()?;
    println!("  Block 1 Out: Min = {:.4}, Max = {:.4}", v2.iter().cloned().fold(f32::INFINITY, f32::min), v2.iter().cloned().fold(f32::NEG_INFINITY, f32::max));

    Ok(())
}
