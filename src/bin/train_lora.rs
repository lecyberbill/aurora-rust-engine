//! LoRA SFT trainer (dense Qwen2 family). CUDA / ROCm / CPU.

use anyhow::Result;
use candle_core::Device;
use std::path::PathBuf;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mut model_dir = None;
    let mut data = None;
    let mut out = None;
    let mut steps = 50usize;
    let mut lr = 2e-4f64;
    let mut rank = 16usize;
    let mut alpha = 32f64;
    let mut max_seq = 256usize;
    let mut val_n = 1usize;
    let mut base_dtype = "f32".to_string();
    let mut device = None;
    let mut targets: Vec<String> = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        .iter()
        .map(|s| s.to_string())
        .collect();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model-dir" => {
                i += 1;
                model_dir = Some(PathBuf::from(&args[i]));
            }
            "--data" => {
                i += 1;
                data = Some(PathBuf::from(&args[i]));
            }
            "--out" => {
                i += 1;
                out = Some(PathBuf::from(&args[i]));
            }
            "--steps" => {
                i += 1;
                steps = args[i].parse()?;
            }
            "--lr" => {
                i += 1;
                lr = args[i].parse()?;
            }
            "--rank" => {
                i += 1;
                rank = args[i].parse()?;
            }
            "--alpha" => {
                i += 1;
                alpha = args[i].parse()?;
            }
            "--max-seq" => {
                i += 1;
                max_seq = args[i].parse()?;
            }
            "--val" => {
                i += 1;
                val_n = args[i].parse()?;
            }
            "--base-dtype" => {
                i += 1;
                base_dtype = args[i].clone();
            }
            "--targets" => {
                i += 1;
                targets = args[i].split(',').map(|s| s.trim().to_string()).collect();
            }
            "--device" => {
                i += 1;
                device = Some(args[i].clone());
            }
            other => anyhow::bail!("unknown arg {other}"),
        }
        i += 1;
    }

    let model_dir = model_dir.unwrap_or_else(|| PathBuf::from("/models/llm/qwen2.5-0.5b-instruct"));
    let data = data.unwrap_or_else(|| PathBuf::from("data/lora_tiny.jsonl"));
    let out = out.unwrap_or_else(|| PathBuf::from("lora_out"));

    let dev = match device.as_deref() {
        Some("cpu") => Device::Cpu,
        #[cfg(feature = "rocm")]
        Some("rocm") => Device::new_rocm(0)?,
        #[cfg(feature = "cuda")]
        Some("cuda") => Device::new_cuda(0)?,
        _ => aurora_rust_engine::device::auto_device()?,
    };
    println!("device: {dev:?}");
    let base_dtype = match base_dtype.as_str() {
        "f16" => candle_core::DType::F16,
        "bf16" => candle_core::DType::BF16,
        _ => candle_core::DType::F32,
    };
    aurora_rust_engine::train::train(
        &model_dir, &data, &out, &dev, rank, alpha, lr, steps, max_seq, &targets, val_n, base_dtype,
    )?;
    Ok(())
}
