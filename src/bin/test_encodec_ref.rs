// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: EnCodec decoder validation vs transformers

use aurora_rust_engine::audio::encodec::EncodecDecoder;
use candle_core::{DType, Device, Tensor};
use safetensors::SafeTensors;

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

fn main() -> anyhow::Result<()> {
    let dev = Device::Cpu;
    let bytes = std::fs::read("outputs/audio_ref/encodec_ref.safetensors")?;
    let st = SafeTensors::deserialize(&bytes)?;
    let codes = load_f32(&st, "codes", &dev)?; // [1,4,50]
    let ref_audio = load_f32(&st, "audio", &dev)?.to_dtype(DType::F32)?;

    let dec = EncodecDecoder::from_musicgen("D:/models/musicgen-small/model.safetensors", &dev, DType::F32)?;
    let mut per_q = Vec::new();
    for i in 0..4 {
        let c: Vec<u32> = codes.narrow(1, i, 1)?.flatten_all()?.to_dtype(DType::U32)?.to_vec1()?;
        per_q.push(Tensor::from_vec(c, (1, 50), &dev)?);
    }
    let audio = dec.decode(&per_q)?;
    let d = (&audio - &ref_audio)?.abs()?;
    println!(
        "[encodec] audio max|diff| = {:.3e}  mean = {:.3e}  shapes {:?} vs {:?}",
        d.max_all()?.to_scalar::<f32>()?,
        d.mean_all()?.to_scalar::<f32>()?,
        audio.dims(),
        ref_audio.dims()
    );
    Ok(())
}
