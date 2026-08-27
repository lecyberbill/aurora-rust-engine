use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use aurora_rust_engine::diffusion::dit::blocks::DoubleStreamBlock;
use std::collections::HashMap;

fn main() -> Result<()> {
    println!("🔬 Differential Test: Rust DoubleStreamBlock 0 vs PyTorch Diffusers");
    let device = Device::Cpu;
    let dtype = DType::F32;

    // 1. Load differential test data
    let diff_data = candle_core::safetensors::load(
        "outputs/flux_showcase/double_block_differential.safetensors",
        &device,
    )?;

    let input_img = diff_data.get("input_img").unwrap();
    let input_txt = diff_data.get("input_txt").unwrap();
    let input_temb = diff_data.get("input_temb").unwrap();
    let input_cos = diff_data.get("input_cos").unwrap();
    let input_sin = diff_data.get("input_sin").unwrap();
    let expected_out_txt = diff_data.get("expected_out_txt").unwrap();
    let expected_out_img = diff_data.get("expected_out_img").unwrap();

    let txt_len = input_txt.dim(1)?;
    let img_len = input_img.dim(1)?;

    let txt_cos = input_cos.narrow(0, 0, txt_len)?;
    let txt_sin = input_sin.narrow(0, 0, txt_len)?;
    let img_cos = input_cos.narrow(0, txt_len, img_len)?;
    let img_sin = input_sin.narrow(0, txt_len, img_seq_len(&diff_data)?)?;

    // 2. Load DoubleStreamBlock 0 from checkpoint
    let ckpt_data = candle_core::safetensors::load(
        r"G:\models\flux\flux2Klein_4b.safetensors",
        &device,
    )?;

    let mut block_tensors = HashMap::new();
    let prefix = "double_blocks.0.";
    let prefix_alt = "model.diffusion_model.double_blocks.0.";

    for (k, v) in ckpt_data.iter() {
        let matched_suffix = if let Some(suffix) = k.strip_prefix(prefix) {
            Some(suffix)
        } else if let Some(suffix) = k.strip_prefix(prefix_alt) {
            Some(suffix)
        } else {
            None
        };

        if let Some(suffix) = matched_suffix {
            block_tensors.insert(suffix.to_string(), v.clone().to_dtype(dtype)?);
        }
    }

    // Attach shared double_stream_modulation
    let img_mod = ckpt_data.get("double_stream_modulation_img.lin.weight")
        .or_else(|| ckpt_data.get("model.diffusion_model.double_stream_modulation_img.lin.weight"))
        .expect("img_mod weight missing")
        .clone()
        .to_dtype(dtype)?;
    block_tensors.insert("img_mod.lin.weight".to_string(), img_mod);

    let txt_mod = ckpt_data.get("double_stream_modulation_txt.lin.weight")
        .or_else(|| ckpt_data.get("model.diffusion_model.double_stream_modulation_txt.lin.weight"))
        .expect("txt_mod weight missing")
        .clone()
        .to_dtype(dtype)?;
    block_tensors.insert("txt_mod.lin.weight".to_string(), txt_mod);

    let vb = VarBuilder::from_tensors(block_tensors, dtype, &device);
    let double_block = DoubleStreamBlock::new(3072, 24, 3, vb)?;

    // 3. Execute Rust forward pass
    let (rust_img, rust_txt) = double_block.forward(
        input_img,
        input_txt,
        input_temb,
        Some(&img_cos),
        Some(&img_sin),
        Some(&txt_cos),
        Some(&txt_sin),
    )?;

    // 4. Compute numerical difference: Diffusers vs Rust
    let img_diff = (&rust_img - expected_out_img)?.abs()?;
    let img_max_diff: f32 = img_diff.flatten_all()?.max(0)?.to_scalar()?;
    let img_mean_diff: f32 = img_diff.flatten_all()?.mean(0)?.to_scalar()?;

    let txt_diff = (&rust_txt - expected_out_txt)?.abs()?;
    let txt_max_diff: f32 = txt_diff.flatten_all()?.max(0)?.to_scalar()?;
    let txt_mean_diff: f32 = txt_diff.flatten_all()?.mean(0)?.to_scalar()?;

    println!("============================================================");
    println!("📊 DoubleStreamBlock 0 Numerical Divergence:");
    println!("   • Image Stream Max Absolute Error:  {:.8}", img_max_diff);
    println!("   • Image Stream Mean Absolute Error: {:.8}", img_mean_diff);
    println!("   • Text Stream  Max Absolute Error:  {:.8}", txt_max_diff);
    println!("   • Text Stream  Mean Absolute Error: {:.8}", txt_mean_diff);
    println!("============================================================");

    if img_max_diff < 0.05 && txt_max_diff < 0.05 {
        println!("✅ PERFECT MATCH! DoubleStreamBlock 0 is numerically iso with PyTorch Diffusers!");
    } else {
        println!("❌ Divergence detected in DoubleStreamBlock!");
    }

    Ok(())
}

fn img_seq_len(diff_data: &HashMap<String, Tensor>) -> Result<usize> {
    Ok(diff_data.get("input_img").unwrap().dim(1)?)
}
