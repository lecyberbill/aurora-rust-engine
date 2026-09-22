// [WFGY] Zone: SAFE | λ: 0.25 | Fallbacks: 0 | Action: Bit-exact validation of the Rust Qwen3 caption encoder vs PyTorch reference

use std::path::Path;

use candle_core::{DType, Device, Tensor};
use safetensors::SafeTensors;
use aurora_rust_engine::text::Qwen3TextEncoder;

fn load_f32(st: &SafeTensors, name: &str, dev: &Device) -> anyhow::Result<Tensor> {
    let t = st.tensor(name)?;
    let shape: Vec<usize> = t.shape().to_vec();
    anyhow::ensure!(t.dtype() == safetensors::Dtype::F32, "{} is not F32", name);
    let data: Vec<f32> = t
        .data()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    Ok(Tensor::from_vec(data, shape, dev)?)
}

fn main() -> anyhow::Result<()> {
    let dev = Device::Cpu;
    let ref_path = "outputs/audio_ref/qwen_ref.safetensors";
    let te_path = Path::new("G:/models/Audio/text_encoder/model.safetensors");
    let tok_path = Path::new("G:/models/Audio/tokenizer/tokenizer.json");

    let buf = std::fs::read(ref_path)?;
    let st = SafeTensors::deserialize(&buf)?;

    let text_ids = load_f32(&st, "text_ids", &dev)?.to_dtype(DType::U32)?;
    let ref_hidden = load_f32(&st, "last_hidden", &dev)?;
    let lyric_ids = load_f32(&st, "lyric_ids", &dev)?.to_dtype(DType::U32)?;
    let ref_lyric = load_f32(&st, "lyric_embeds", &dev)?;

    let model = Qwen3TextEncoder::from_safetensors(te_path, Some(tok_path), &dev, DType::F32)?;

    let hidden = model.forward_last_hidden(&text_ids)?;
    let lyric = model.embed_ids(&lyric_ids)?;

    let d_hidden = (hidden.sub(&ref_hidden)?.abs()?.max_all()?.to_scalar::<f32>())?;
    let d_lyric = (lyric.sub(&ref_lyric)?.abs()?.max_all()?.to_scalar::<f32>())?;
    println!("=== Qwen3 caption encoder ===");
    println!("  ids shape = {:?}", text_ids.shape());
    println!("  last_hidden shape = {:?}, ref = {:?}", hidden.shape(), ref_hidden.shape());
    println!("  last_hidden max|diff| = {:.3e}", d_hidden);
    println!("  lyric_embed max|diff| = {:.3e}", d_lyric);
    println!("  ref  hidden mean = {:.6}", ref_hidden.mean_all()?.to_scalar::<f32>()?);

    Ok(())
}
