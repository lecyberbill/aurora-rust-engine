// [WFGY] Zone: SAFE | λ: 0.15 | Fallbacks: 0 | Action: Debug Single Block forward pass to locate exact NaN source

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use aurora_rust_engine::weights::{SafeTensorsArchive, WeightRouter};
use aurora_rust_engine::diffusion::dit::blocks::DoubleStreamBlock;

fn main() -> anyhow::Result<()> {
    println!("🔍 Testing Single DoubleStreamBlock.0 Forward Pass...");

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

    let img = Tensor::randn(0f32, 1f32, (1, 1024, 3072), &device)?.to_dtype(DType::F16)?;
    let txt = Tensor::randn(0f32, 1f32, (1, 256, 3072), &device)?.to_dtype(DType::F16)?;
    let temb = Tensor::randn(0f32, 1f32, (1, 3072), &device)?.to_dtype(DType::F16)?;

    println!("⚡ Executing Block.0 via streamer...");
    let (next_img, next_txt) = streamer.execute_double_block(0, &img, &txt, &temb, None, None)?;

    let img_vec = next_img.flatten_all()?.to_dtype(DType::F32)?.to_device(&Device::Cpu)?.to_vec1::<f32>()?;
    let min_v = img_vec.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_v = img_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean_v = img_vec.iter().sum::<f32>() / img_vec.len() as f32;

    println!("📊 Next Img Stats: Min = {:.4}, Max = {:.4}, Mean = {:.4}", min_v, max_v, mean_v);

    Ok(())
}
