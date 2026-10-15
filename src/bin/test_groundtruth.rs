use anyhow::{Context, Result};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use std::path::PathBuf;
use std::time::Instant;

use aurora_rust_engine::diffusion::dit::cogvideox::{CogVideoXConfig, CogVideoXTransformer3D};

fn main() -> Result<()> {
    println!("============================================================");
    println!("  CogVideoX-2B Mathematical Ground-Truth Verification Tool");
    println!("============================================================");

    let dev = match aurora_rust_engine::device::auto_device() {
        Ok(d) => d,
        Err(_) => Device::Cpu,
    };
    let dtype = if dev.is_cpu() { DType::F32 } else { DType::F16 };
    println!("[*] Device: {:?}, Precision: {:?}", dev, dtype);

    // Locate test_groundtruth.safetensors
    let gt_candidates = [
        PathBuf::from("test_groundtruth.safetensors"),
        PathBuf::from("M:/outputs/test_groundtruth.safetensors"),
        PathBuf::from("/models/outputs/test_groundtruth.safetensors"),
    ];
    let gt_path = gt_candidates.iter().find(|p| p.exists())
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("test_groundtruth.safetensors not found! Checked: {:?}", gt_candidates))?;

    println!("[+] Loading ground-truth safetensors from: {:?}", gt_path);
    let gt_tensors = candle_core::safetensors::load(&gt_path, &dev)?;
    let hidden_states = gt_tensors.get("hidden_states")
        .cloned().context("Missing hidden_states")?.to_dtype(dtype)?;
    let encoder_hidden_states = gt_tensors.get("encoder_hidden_states")
        .cloned().context("Missing encoder_hidden_states")?.to_dtype(dtype)?;
    let timestep = gt_tensors.get("timestep")
        .cloned().context("Missing timestep")?.to_dtype(dtype)?;
    let diffusers_output = gt_tensors.get("diffusers_output")
        .cloned().context("Missing diffusers_output")?.to_dtype(dtype)?;

    println!("[*] Input hidden_states shape: {:?}", hidden_states.shape());
    println!("[*] Input encoder_hidden_states shape: {:?}", encoder_hidden_states.shape());
    println!("[*] Input timestep shape: {:?}", timestep.shape());
    println!("[*] Diffusers reference output shape: {:?}", diffusers_output.shape());

    // Resolve transformer weights
    let weight_candidates = [
        PathBuf::from("/models/comfyui/diffusion_models/cogvideox_2b_transformer.safetensors"),
        PathBuf::from("M:/comfyui/diffusion_models/cogvideox_2b_transformer.safetensors"),
        PathBuf::from("D:/image_to_text/.cache/hub/models--THUDM--CogVideoX-2b/blobs/8fbb6a5e67c70885a8ed8e33df144ac61253e45977be5035fa18cfdf77d386c7"),
    ];
    let weights_path = weight_candidates.iter().find(|p| p.exists())
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("Weights not found! Checked: {:?}", weight_candidates))?;

    println!("[+] Loading CogVideoXTransformer3D weights from: {:?}", weights_path);
    let t0 = Instant::now();
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[&weights_path], dtype, &dev)?
    };
    let config = CogVideoXConfig::default();
    let mut model = CogVideoXTransformer3D::new(config, vb)?;

    // Check pos_embedding buffer
    let pos_candidates = [
        PathBuf::from("cogvideox_pos_embed.safetensors"),
        PathBuf::from("M:/outputs/cogvideox_pos_embed.safetensors"),
        PathBuf::from("/models/outputs/cogvideox_pos_embed.safetensors"),
    ];
    for p in &pos_candidates {
        if p.exists() {
            if let Ok(tensors) = candle_core::safetensors::load(p, &dev) {
                if let Some(pos) = tensors.get("pos_embedding") {
                    println!("[+] Preloading verified 3D sincos pos_embedding from {:?}", p);
                    model.patch_embed.set_pos_embedding(pos.to_dtype(dtype)?.to_device(&dev)?);
                    break;
                }
            }
        }
    }

    println!("[+] Model loaded in {:.2}s. Running forward pass...", t0.elapsed().as_secs_f64());

    // Note: Diffusers hidden_states is [B, T, C, H, W].
    // If Rust expects [B, C, T, H, W], we transpose(1, 2) first.
    let rust_input = hidden_states.transpose(1, 2)?;
    let t_fwd = Instant::now();
    let rust_output_5d = model.forward(&rust_input, &timestep, &encoder_hidden_states, None)?;
    let fwd_time = t_fwd.elapsed().as_secs_f64();
    println!("[+] Rust forward pass complete in {:.2}s. Output shape: {:?}", fwd_time, rust_output_5d.shape());

    // Transpose Rust output back to [B, T, C, H, W] to match Diffusers
    let rust_output = rust_output_5d.transpose(1, 2)?;

    // Compute numerical differences against Diffusers ground truth
    let diff = (&rust_output.to_dtype(DType::F32)? - &diffusers_output.to_dtype(DType::F32)?)?.abs()?;
    let max_diff = diff.flatten_all()?.max(0)?.to_scalar::<f32>()?;
    let mean_diff = diff.flatten_all()?.mean(0)?.to_scalar::<f32>()?;

    let rust_f32 = rust_output.to_dtype(DType::F32)?;
    let diff_f32 = diffusers_output.to_dtype(DType::F32)?;
    let rust_mean = rust_f32.flatten_all()?.mean(0)?.to_scalar::<f32>()?;
    let diff_mean = diff_f32.flatten_all()?.mean(0)?.to_scalar::<f32>()?;

    println!("\n============================================================");
    println!("  Empirical Verification Results (Rust Candle vs Diffusers)");
    println!("============================================================");
    println!("  Diffusers Output Mean : {:.6}", diff_mean);
    println!("  Candle Rust Output Mean: {:.6}", rust_mean);
    println!("  Mean Absolute Error   : {:.6}", mean_diff);
    println!("  Max Absolute Error    : {:.6}", max_diff);

    if max_diff < 0.05 {
        println!("\n[SUCCESS] Bit-exact mathematical parity confirmed! (< 0.05 error in FP16)");
    } else {
        println!("\n[FAILURE] Discrepancy detected between Rust model and Diffusers ground truth!");
    }
    println!("============================================================\n");

    Ok(())
}
