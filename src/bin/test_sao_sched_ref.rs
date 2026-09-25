// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: CosineDPM scheduler validation vs diffusers

use aurora_rust_engine::models::CosineDpmScheduler;
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
    let bytes = std::fs::read("outputs/audio_ref/sao_sched.safetensors")?;
    let st = SafeTensors::deserialize(&bytes)?;
    let noise = load_f32(&st, "noise", &dev)?.to_dtype(DType::F32)?;
    let mut sample = load_f32(&st, "sample0", &dev)?.to_dtype(DType::F32)?;

    let mut sched = CosineDpmScheduler::new(8);
    let ref_sigmas: Vec<f32> = load_f32(&st, "sigmas", &dev)?.to_vec1()?;
    let sig_d = sched
        .sigmas
        .iter()
        .zip(ref_sigmas.iter())
        .map(|(a, b)| (a - *b as f64).abs())
        .fold(0f64, f64::max);
    println!("sigmas max|diff| = {:.3e}", sig_d);

    for i in 0..8 {
        let mo = load_f32(&st, &format!("mo{i}"), &dev)?.to_dtype(DType::F32)?;
        sample = sched.step(&mo, &sample, i, &noise)?;
        let ref_prev = load_f32(&st, &format!("prev{i}"), &dev)?.to_dtype(DType::F32)?;
        let d = (&sample - &ref_prev)?.abs()?;
        println!(
            "step {i}: prev max|diff| = {:.3e}  mean = {:.3e}",
            d.max_all()?.to_scalar::<f32>()?,
            d.mean_all()?.to_scalar::<f32>()?
        );
    }
    Ok(())
}
