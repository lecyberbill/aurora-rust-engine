// [WFGY] Zone: SAFE | λ: 0.15 | Fallbacks: 0 | Action: Debug inside DoubleStreamBlock 3 step-by-step

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use aurora_rust_engine::weights::SafeTensorsArchive;
use aurora_rust_engine::diffusion::dit::blocks::DoubleStreamBlock;
use std::collections::HashMap;

fn main() -> anyhow::Result<()> {
    println!("🔍 Testing DoubleStreamBlock 2 vs Block 3...");

    let device = Device::new_cuda(0)?;
    let checkpoint = "G:\\models\\flux\\flux1SchnellFp8_schnellFp8.safetensors";
    let archive = SafeTensorsArchive::open(checkpoint)?;

    let mut load_block = |idx: usize| -> anyhow::Result<DoubleStreamBlock> {
        let mut tensors = HashMap::new();
        let prefix = format!("model.diffusion_model.double_blocks.{}.", idx);
        for key in archive.keys() {
            if let Some(s) = key.strip_prefix(&prefix) {
                tensors.insert(s.to_string(), archive.get_tensor(key, &device, DType::F16)?);
            }
        }
        let vb = VarBuilder::from_tensors(tensors, DType::F16, &device);
        Ok(DoubleStreamBlock::new(3072, 24, 4, vb)?)
    };

    let block0 = load_block(0)?;
    let block1 = load_block(1)?;
    let block2 = load_block(2)?;
    let block3 = load_block(3)?;

    let img0 = Tensor::randn(0f32, 1f32, (1, 1024, 3072), &device)?.to_dtype(DType::F16)?;
    let txt0 = Tensor::randn(0f32, 1f32, (1, 256, 3072), &device)?.to_dtype(DType::F16)?;
    let temb = Tensor::randn(0f32, 1f32, (1, 3072), &device)?.to_dtype(DType::F16)?;

    let (img1, txt1) = block0.forward(&img0, &txt0, &temb, None, None)?;
    let (img2, txt2) = block1.forward(&img1, &txt1, &temb, None, None)?;
    let (img3, txt3) = block2.forward(&img2, &txt2, &temb, None, None)?;

    let v3 = img3.flatten_all()?.to_dtype(DType::F32)?.to_device(&Device::Cpu)?.to_vec1::<f32>()?;
    println!("  Block 2 Out: Min = {:.4}, Max = {:.4}", v3.iter().cloned().fold(f32::INFINITY, f32::min), v3.iter().cloned().fold(f32::NEG_INFINITY, f32::max));

    let (img4, txt4) = block3.forward(&img3, &txt3, &temb, None, None)?;
    let v4 = img4.flatten_all()?.to_dtype(DType::F32)?.to_device(&Device::Cpu)?.to_vec1::<f32>()?;
    println!("  Block 3 Out: Min = {:.4}, Max = {:.4}", v4.iter().cloned().fold(f32::INFINITY, f32::min), v4.iter().cloned().fold(f32::NEG_INFINITY, f32::max));

    Ok(())
}
