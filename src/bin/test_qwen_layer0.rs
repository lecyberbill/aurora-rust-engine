// [WFGY] Zone: SAFE | λ: 0.10 | Fallbacks: 0 | Action: Pinpoint exact difference between inline and QwenDecoderLayer

use candle_core::{DType, Device, Module, Tensor};
use aurora_rust_engine::weights::SafeTensorsArchive;
use aurora_rust_engine::text::Qwen3TextEncoder;
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let p = "/models/comfyui/text_encoders/qwen3vl_4b_fp8_scaled.safetensors";
    let archive = SafeTensorsArchive::open(p)?;
    let dev = Device::Cpu;
    let tok_path = PathBuf::from("qwen_tokenizer_KREA2.json");

    let encoder = Qwen3TextEncoder::from_archive(&archive, Some(&tok_path), &dev, DType::F32)?;
    let tok = tokenizers::Tokenizer::from_file(&tok_path).unwrap();
    let prompt = "a majestic white wolf on a snowy cliff at golden hour, photorealistic, 8k";
    let formatted_prompt = format!(
        "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        prompt
    );
    let enc = tok.encode(formatted_prompt.as_str(), true).unwrap();
    let ids: Vec<u32> = enc.get_ids().to_vec();
    let seq_len = ids.len();
    let ids_tensor = Tensor::from_vec(ids, (1, seq_len), &dev)?;

    // We can directly test layer 0 of the encoder:
    // Access embed_tokens
    println!("Step 1: embed_tokens...");
    // Let's run full encoder with debug prints
    let taps = encoder.encode_krea2_12_layers(prompt, 0)?;
    println!("Taps dims: {:?}", taps.dims());

    Ok(())
}
