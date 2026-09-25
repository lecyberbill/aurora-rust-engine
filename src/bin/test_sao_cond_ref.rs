// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: Stable Audio number-conditioner validation vs diffusers

use aurora_rust_engine::models::StableAudioProjectionModel;
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
    let ref_path = "outputs/audio_ref/sao_ref.safetensors";
    let bytes = std::fs::read(ref_path)?;
    let st = SafeTensors::deserialize(&bytes)?;
    let ref_start = load_f32(&st, "start_hidden", &dev)?.to_dtype(DType::F32)?; // [1,1,768]
    let ref_end = load_f32(&st, "end_hidden", &dev)?.to_dtype(DType::F32)?;

    let model = StableAudioProjectionModel::from_safetensors(
        "G:/models/Audio/stable-audio-open-models/projection_model/diffusion_pytorch_model.safetensors",
        &dev,
        DType::F32,
    )?;

    let start = Tensor::new(&[0.0f32], &dev)?;
    let end = Tensor::new(&[10.0f32], &dev)?;
    let out = model.forward(None, Some(&start), Some(&end))?;
    let s = out.seconds_start_hidden_states.unwrap();
    let e = out.seconds_end_hidden_states.unwrap();

    let ds = (&s - &ref_start)?.abs()?.max_all()?.to_scalar::<f32>()?;
    let de = (&e - &ref_end)?.abs()?.max_all()?.to_scalar::<f32>()?;
    println!("[sao-cond] start_hidden max|diff| = {:.3e}", ds);
    println!("[sao-cond] end_hidden   max|diff| = {:.3e}", de);
    println!("[sao-cond] shapes rust {:?} {:?} / ref {:?} {:?}", s.dims(), e.dims(), ref_start.dims(), ref_end.dims());
    Ok(())
}
