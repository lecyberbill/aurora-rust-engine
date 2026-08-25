// [WFGY] Zone: SAFE | λ: 0.15 | Fallbacks: 0 | Action: Instrument DoubleStreamBlock 2 forward pass to find divergence

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use aurora_rust_engine::weights::SafeTensorsArchive;
use aurora_rust_engine::diffusion::dit::blocks::DoubleStreamBlock;
use std::collections::HashMap;

fn main() -> anyhow::Result<()> {
    println!("🔍 Testing DoubleStreamBlock 2 vs Block 3 with realistic inputs...");

    let device = Device::new_cuda(0)?;
    let checkpoint = "G:\\models\\flux\\flux1SchnellFp8_schnellFp8.safetensors";
    let archive = SafeTensorsArchive::open(checkpoint)?;

    let load_block = |idx: usize| -> anyhow::Result<DoubleStreamBlock> {
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

    // Start with small realistic variance
    let img0 = (Tensor::randn(0f32, 1f32, (1, 1024, 3072), &device)? * 0.1)?.to_dtype(DType::F16)?;
    let txt0 = (Tensor::randn(0f32, 1f32, (1, 256, 3072), &device)? * 0.1)?.to_dtype(DType::F16)?;
    let temb = (Tensor::randn(0f32, 1f32, (1, 3072), &device)? * 0.1)?.to_dtype(DType::F16)?;

    let print_stats = |name: &str, t: &Tensor| -> anyhow::Result<()> {
        let v = t.flatten_all()?.to_dtype(DType::F32)?.to_device(&Device::Cpu)?.to_vec1::<f32>()?;
        let min = v.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let has_nan = v.iter().any(|x| x.is_nan());
        println!("  {}: Min = {:.4}, Max = {:.4}, Has NaN = {}", name, min, max, has_nan);
        Ok(())
    };

    let (img1, txt1) = block0.forward(&img0, &txt0, &temb, None, None)?;
    print_stats("Block 0 Out (img)", &img1)?;

    let (img2, txt2) = block1.forward(&img1, &txt1, &temb, None, None)?;
    print_stats("Block 1 Out (img)", &img2)?;

    let (img3, txt3) = block2.forward(&img2, &txt2, &temb, None, None)?;
    print_stats("Block 2 Out (img)", &img3)?;

    let (img4, txt4) = block3.forward(&img3, &txt3, &temb, None, None)?;
    print_stats("Block 3 Out (img)", &img4)?;

    Ok(())
}
