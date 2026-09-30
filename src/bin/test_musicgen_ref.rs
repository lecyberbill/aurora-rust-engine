// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: MusicGen decoder validation vs transformers

use aurora_rust_engine::models::MusicgenDecoder;
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
    let bytes = std::fs::read("outputs/audio_ref/musicgen_ref.safetensors")?;
    let st = SafeTensors::deserialize(&bytes)?;
    let ids = load_f32(&st, "ids", &dev)?.to_dtype(DType::F32)?; // [4,8]
    let enc = load_f32(&st, "enc", &dev)?.to_dtype(DType::F32)?; // [1,20,1024]
    let ref_logits = load_f32(&st, "logits", &dev)?.to_dtype(DType::F32)?; // [4,8,2048]

    // reshape ids to [1,4,8] (u32)
    let ids_u: Vec<u32> = ids.flatten_all()?.to_dtype(DType::U32)?.to_vec1()?;
    let ids3 = Tensor::from_vec(ids_u, (1, 4, 8), &dev)?;

    let dec = MusicgenDecoder::from_musicgen("D:/models/musicgen-small/model.safetensors", &dev, DType::F32)?;
    let logits = dec.forward(&ids3, &enc)?.reshape((4, 8, 2048))?;
    let d = (&logits - &ref_logits)?.abs()?;
    println!(
        "[musicgen] logits max|diff| = {:.3e}  mean = {:.3e}  shapes {:?} vs {:?}",
        d.max_all()?.to_scalar::<f32>()?,
        d.mean_all()?.to_scalar::<f32>()?,
        logits.dims(),
        ref_logits.dims()
    );
    Ok(())
}
