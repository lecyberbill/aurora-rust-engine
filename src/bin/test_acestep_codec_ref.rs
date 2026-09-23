// [WFGY] Zone: SAFE | λ: 0.15 | Fallbacks: 0 | Action: Validate the Rust 5Hz audio codec (FSQ tokenizer + detokenizer) vs PyTorch

use candle_core::{DType, Device, Tensor};
use safetensors::SafeTensors;
use aurora_rust_engine::models::AceStepAudioCodec;

fn load_f32(st: &SafeTensors, name: &str, dev: &Device) -> anyhow::Result<Tensor> {
    let t = st.tensor(name)?;
    let shape: Vec<usize> = t.shape().to_vec();
    let data: Vec<f32> = t
        .data()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    Ok(Tensor::from_vec(data, shape, dev)?)
}

fn max_abs(a: &Tensor, b: &Tensor) -> anyhow::Result<f32> {
    Ok((a - b)?.abs()?.max_all()?.to_scalar::<f32>()?)
}

fn main() -> anyhow::Result<()> {
    let dev = Device::Cpu;
    let buf = std::fs::read("outputs/audio_ref/codec_ref.safetensors")?;
    let st = SafeTensors::deserialize(&buf)?;
    let features = load_f32(&st, "features", &dev)?; // [1, T, 64]
    let ref_q = load_f32(&st, "quantized", &dev)?; // [1, Tp, 2048]
    let ref_idx = load_f32(&st, "indices", &dev)?; // [1, Tp, 1]
    let ref_detok = load_f32(&st, "detok", &dev)?; // [1, Tp*5, 64]
    let ref_from_idx = load_f32(&st, "from_idx", &dev)?; // [1, Tp, 2048]
    let ref_detok_idx = load_f32(&st, "detok_from_idx", &dev)?;

    let codec = AceStepAudioCodec::from_safetensors(
        "G:/models/Audio-base/audio_codec/codec.safetensors",
        &dev,
        DType::F32,
    )?;

    let (q, idx) = codec.tokenize(&features)?;
    println!("=== Tokenizer ===");
    println!("  quantized r={:?} ref={:?} max|diff|={:.3e}", q.shape(), ref_q.shape(), max_abs(&q, &ref_q)?);
    let idx_f = idx.to_dtype(DType::F32)?.unsqueeze(2)?;
    println!("  indices   r={:?} ref={:?} max|diff|={:.3e}", idx_f.shape(), ref_idx.shape(), max_abs(&idx_f, &ref_idx)?);
    println!("  idx rust={:?}", idx.flatten_all()?.to_vec1::<u32>()?);

    let detok = codec.detokenize(&q)?;
    println!("=== Detokenizer ===");
    println!("  detok(quantized)   r={:?} ref={:?} max|diff|={:.3e}", detok.shape(), ref_detok.shape(), max_abs(&detok, &ref_detok)?);

    let from_idx = codec.tokenizer.quantizer.get_output_from_indices(&idx)?;
    println!("  fsq index->tokens  r={:?} ref={:?} max|diff|={:.3e}", from_idx.shape(), ref_from_idx.shape(), max_abs(&from_idx, &ref_from_idx)?);
    let detok_idx = codec.detokenize_from_indices(&idx)?;
    println!("  detok(from_idx)    r={:?} ref={:?} max|diff|={:.3e}", detok_idx.shape(), ref_detok_idx.shape(), max_abs(&detok_idx, &ref_detok_idx)?);

    Ok(())
}
