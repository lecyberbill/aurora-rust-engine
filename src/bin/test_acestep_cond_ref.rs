// [WFGY] Zone: SAFE | λ: 0.25 | Fallbacks: 0 | Action: Bit-exact validation of the Rust ACE-Step ConditionEncoder vs PyTorch reference

use candle_core::{DType, Device, Tensor};
use safetensors::SafeTensors;
use aurora_rust_engine::models::AceStepConditionEncoder;

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

fn max_abs_diff(a: &Tensor, b: &Tensor) -> anyhow::Result<f32> {
    Ok((a - b)?.abs()?.max_all()?.to_scalar::<f32>()?)
}

fn main() -> anyhow::Result<()> {
    let dev = Device::Cpu;
    let ref_path = "outputs/audio_ref/cond_ref.safetensors";
    let cond_path = "G:/models/Audio/condition_encoder/diffusion_pytorch_model.safetensors";

    let buf = std::fs::read(ref_path)?;
    let st = SafeTensors::deserialize(&buf)?;

    let text_hidden = load_f32(&st, "text_hidden", &dev)?;
    let lyric_embeds = load_f32(&st, "lyric_embeds", &dev)?;
    let ref_text = load_f32(&st, "text_out", &dev)?;
    let ref_lyric = load_f32(&st, "lyric_out", &dev)?;
    let ref_timbre = load_f32(&st, "timbre_out", &dev)?;
    let ref_cond = load_f32(&st, "condition", &dev)?;

    let model = AceStepConditionEncoder::from_safetensors(cond_path, &dev, DType::F32)?;

    println!("=== Condition encoder ===");
    println!("  text   max|diff| = {:.3e}", max_abs_diff(&model.forward_text(&text_hidden)?, &ref_text)?);
    println!("  lyric  max|diff| = {:.3e}", max_abs_diff(&model.forward_lyrics(&lyric_embeds)?, &ref_lyric)?);
    let ref_lat = model.silence_latent.narrow(1, 0, AceStepConditionEncoder::TIMBRE_FIX_FRAME)?;
    println!("  timbre max|diff| = {:.3e}", max_abs_diff(&model.encode_timbre(&ref_lat)?, &ref_timbre)?);
    let cond = model.forward_condition(&text_hidden, &lyric_embeds)?;
    println!("  rust cond shape = {:?}, ref = {:?}", cond.shape(), ref_cond.shape());
    println!("  cond   max|diff| = {:.3e}", max_abs_diff(&cond, &ref_cond)?);
    println!("  ref cond mean = {:.6}", ref_cond.mean_all()?.to_scalar::<f32>()?);

    Ok(())
}
