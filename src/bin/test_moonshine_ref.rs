// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: Moonshine validation vs transformers

use aurora_rust_engine::models::MoonshineModel;
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
    let bytes = std::fs::read("outputs/audio_ref/moonshine_ref.safetensors")?;
    let st = SafeTensors::deserialize(&bytes)?;
    let input = load_f32(&st, "input_values", &dev)?.to_dtype(DType::F32)?;
    let ref_enc = load_f32(&st, "enc", &dev)?.to_dtype(DType::F32)?;
    let ref_ids: Vec<f32> = load_f32(&st, "ids", &dev)?.flatten_all()?.to_vec1()?;

    let model = MoonshineModel::from_safetensors(
        "G:/models/moonshine-tiny/model.safetensors",
        &dev,
        DType::F32,
    )?;
    let enc = model.encode(&input)?;
    let d = (&enc - &ref_enc)?.abs()?;
    println!(
        "[moonshine] enc max|diff| = {:.3e}  mean = {:.3e}  shapes {:?} vs {:?}",
        d.max_all()?.to_scalar::<f32>()?,
        d.mean_all()?.to_scalar::<f32>()?,
        enc.dims(),
        ref_enc.dims()
    );

    let ids = model.greedy_decode(&ref_enc, 1, 2, 80)?;
    let ref_ids_u: Vec<u32> = ref_ids.iter().map(|&x| x as u32).collect();
    println!("[moonshine] ids rust = {:?}", ids);
    println!("[moonshine] ids ref  = {:?}", ref_ids_u);
    println!("[moonshine] ids match = {}", ids == ref_ids_u);
    Ok(())
}
