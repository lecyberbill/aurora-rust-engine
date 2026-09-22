// [WFGY] Zone: SAFE | λ: 0.10 | Fallbacks: 0 | Action: Validate APG guidance against PyTorch reference

use candle_core::{DType, Device, Tensor};
use safetensors::SafeTensors;
use aurora_rust_engine::models::acestep::apg_forward;

fn load(st: &SafeTensors, name: &str, dev: &Device) -> anyhow::Result<Tensor> {
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
    let buf = std::fs::read("outputs/audio_ref/apg_ref.safetensors")?;
    let st = SafeTensors::deserialize(&buf)?;
    let pc = load(&st, "pc", &dev)?; // [4,1,64,8]
    let pu = load(&st, "pu", &dev)?;
    let out = load(&st, "out", &dev)?;
    let n = pc.dim(0)?;

    let mut momentum: Option<Tensor> = None;
    let mut max_diff = 0f32;
    for i in 0..n {
        let a = pc.narrow(0, i, 1)?.squeeze(0)?; // [1,64,8]
        let b = pu.narrow(0, i, 1)?.squeeze(0)?;
        let r = apg_forward(&a, &b, 7.0, &mut momentum)?;
        let ref_i = out.narrow(0, i, 1)?.squeeze(0)?;
        let d = (r.sub(&ref_i)?.abs()?.max_all()?.to_scalar::<f32>())?;
        max_diff = max_diff.max(d);
        println!("  step {} max|diff| = {:.3e}", i, d);
    }
    println!("APG overall max|diff| = {:.3e}", max_diff);
    let _ = DType::F32;
    Ok(())
}
