//! Generate with a dense model, optionally applying a PEFT LoRA adapter.

use anyhow::Result;
use candle_core::Device;
use std::path::PathBuf;
use tokenizers::Tokenizer;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mut model_dir = PathBuf::from("/models/llm/qwen2.5-0.5b-instruct");
    let mut adapter: Option<PathBuf> = None;
    let mut prompt = String::from("What is the magic word?");
    let mut max_new = 32usize;
    let mut base_dtype = "f32".to_string();
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
                adapter = Some(PathBuf::from(&args[i]));
            }
            "--prompt" => {
                i += 1;
                prompt = args[i].clone();
            }
            "--max-new" => {
                i += 1;
                max_new = args[i].parse()?;
            }
            "--base-dtype" => {
                i += 1;
                base_dtype = args[i].clone();
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

    let tok = Tokenizer::from_file(model_dir.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let bdt = match base_dtype.as_str() {
        "f16" => candle_core::DType::F16,
        "bf16" => candle_core::DType::BF16,
        _ => candle_core::DType::F32,
    };
    let model = match &adapter {
        Some(a) => aurora_rust_engine::train::load_with_adapter(&model_dir, a, &dev, bdt)?,
        None => aurora_rust_engine::train::LoraModel::load_with_dtype(&model_dir, &dev, 0, 1.0, &[], bdt)?,
    };
    let chat = format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n");
    print!("{}", model.generate(&tok, &chat, max_new)?);
    Ok(())
}
