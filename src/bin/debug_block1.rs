// [WFGY] Zone: SAFE | λ: 0.15 | Fallbacks: 0 | Action: Debug inside DoubleStreamBlock 1 layer by layer

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use aurora_rust_engine::weights::{SafeTensorsArchive, WeightRouter};
use aurora_rust_engine::diffusion::dit::blocks::DoubleStreamBlock;

fn main() -> anyhow::Result<()> {
    println!("🔍 Testing DoubleStreamBlock 1 internal tensor statistics...");

    let device = Device::new_cuda(0)?;
    let checkpoint = "G:\\models\\flux\\flux1SchnellFp8_schnellFp8.safetensors";
    let archive = std::sync::Arc::new(SafeTensorsArchive::open(checkpoint)?);
    let streamer = aurora_rust_engine::diffusion::dit::streamer::SequentialBlockStreamer::new(
        archive,
        device.clone(),
        DType::F16,
        3072,
        24,
        4,
    );

    let img0 = Tensor::randn(0f32, 1f32, (1, 1024, 3072), &device)?.to_dtype(DType::F16)?;
    let txt0 = Tensor::randn(0f32, 1f32, (1, 256, 3072), &device)?.to_dtype(DType::F16)?;
    let temb = Tensor::randn(0f32, 1f32, (1, 3072), &device)?.to_dtype(DType::F16)?;

    let (img1, txt1) = streamer.execute_double_block(0, &img0, &txt0, &temb, None, None)?;
    let v1 = img1.flatten_all()?.to_dtype(DType::F32)?.to_device(&Device::Cpu)?.to_vec1::<f32>()?;
    let min1 = v1.iter().cloned().fold(f32::INFINITY, f32::min);
    let max1 = v1.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!("📊 Output Block 0: Min = {:.4}, Max = {:.4}", min1, max1);

    // Now test Block 1
    let (img2, txt2) = streamer.execute_double_block(1, &img1, &txt1, &temb, None, None)?;
    let v2 = img2.flatten_all()?.to_dtype(DType::F32)?.to_device(&Device::Cpu)?.to_vec1::<f32>()?;
    let min2 = v2.iter().cloned().fold(f32::INFINITY, f32::min);
    let max2 = v2.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!("📊 Output Block 1: Min = {:.4}, Max = {:.4}", min2, max2);

    Ok(())
}
