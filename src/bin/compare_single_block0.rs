use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use aurora_rust_engine::diffusion::dit::blocks::SingleStreamBlock;
use std::collections::HashMap;

fn main() -> Result<()> {
    println!("🔬 Differential Test: Rust SingleStreamBlock 0 vs PyTorch Diffusers");

    let device = Device::Cpu;
    let dtype = DType::F32;

    // 1. Load the generated reference tensors
    let diff_data = candle_core::safetensors::load(
        "outputs/flux_showcase/single_block_differential.safetensors",
        &device,
    )?;

    let input_x = diff_data.get("input_x").expect("input_x missing").to_dtype(dtype)?;
    let input_temb = diff_data.get("input_temb").expect("input_temb missing").to_dtype(dtype)?;
    let input_cos = diff_data.get("input_cos").expect("input_cos missing").to_dtype(dtype)?;
    let input_sin = diff_data.get("input_sin").expect("input_sin missing").to_dtype(dtype)?;
    let expected_single_out = diff_data
        .get("expected_single_out")
        .expect("expected_single_out missing")
        .to_dtype(dtype)?;

    // 2. Load model weights for SingleStreamBlock 0 from checkpoint
    let ckpt_data = candle_core::safetensors::load(
        r"G:\models\flux\flux2Klein_4b.safetensors",
        &device,
    )?;

    let mut block_tensors = HashMap::new();
    let prefix = "single_blocks.0.";
    let prefix_alt = "model.diffusion_model.single_blocks.0.";

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

    // Also attach single_stream_modulation
    let mod_tensor = ckpt_data.get("single_stream_modulation.lin.weight")
        .or_else(|| ckpt_data.get("model.diffusion_model.single_stream_modulation.lin.weight"))
        .expect("single_stream_modulation weight missing")
        .clone()
        .to_dtype(dtype)?;
    block_tensors.insert("modulation.lin.weight".to_string(), mod_tensor);

    let vb = VarBuilder::from_tensors(block_tensors, dtype, &device);
    let single_block = SingleStreamBlock::new(3072, 24, 3, vb)?;

    // 3. Execute Rust forward pass
    let rust_out = single_block.forward(
        &input_x,
        &input_temb,
        Some(&input_cos),
        Some(&input_sin),
    )?;

    // 4. Compute numerical difference: Diffusers vs Rust
    let diff = (&rust_out - &expected_single_out)?.abs()?;
    let max_diff: f32 = diff.flatten_all()?.max(0)?.to_scalar()?;
    let mean_diff: f32 = diff.flatten_all()?.mean(0)?.to_scalar()?;

    println!("============================================================");
    println!("📊 SingleStreamBlock 0 Numerical Divergence:");
    println!("   • Max Absolute Error:  {:.8}", max_diff);
    println!("   • Mean Absolute Error: {:.8}", mean_diff);
    println!("============================================================");

    if max_diff < 1e-4 {
        println!("✅ SingleStreamBlock is 100% IDENTICAL to PyTorch Diffusers!");
    } else {
        println!("❌ Divergence detected! Let's inspect sub-operations...");
    }

    Ok(())
}
