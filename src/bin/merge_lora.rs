//! Merge a PEFT LoRA adapter into a dense model (standalone model dir).

use anyhow::Result;
use candle_core::{DType, Device};
use std::path::PathBuf;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mut model_dir = PathBuf::from("/models/llm/qwen2.5-0.5b-instruct");
    let mut adapter = PathBuf::from("/tmp/lora_out");
    let mut out = PathBuf::from("/tmp/merged");
    let mut dtype = DType::F16;
    let mut device = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model-dir" => {
                i += 1;
                model_dir = PathBuf::from(&args[i]);
            }
            "--adapter-dir" => {
                i += 1;
                adapter = PathBuf::from(&args[i]);
            }
            "--out" => {
                i += 1;
                out = PathBuf::from(&args[i]);
            }
            "--dtype" => {
                i += 1;
                dtype = match args[i].as_str() {
                    "f16" => DType::F16,
                    "f32" => DType::F32,
                    other => anyhow::bail!("dtype {other} (f16|f32)"),
                };
            }
            "--device" => {
                i += 1;
                device = Some(args[i].clone());
            }
            other => anyhow::bail!("unknown arg {other}"),
        }
        i += 1;
    }

    let dev = match device.as_deref() {
        Some("cpu") => Device::Cpu,
        #[cfg(feature = "rocm")]
        Some("rocm") => Device::new_rocm(0)?,
        #[cfg(feature = "cuda")]
        Some("cuda") => Device::new_cuda(0)?,
        _ => aurora_rust_engine::device::auto_device()?,
    };
    aurora_rust_engine::train::merge_adapter(&model_dir, &adapter, &out, &dev, dtype)?;
    Ok(())
}
