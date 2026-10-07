use aurora_rust_engine::weights::SafeTensorsArchive;
use candle_core::{DType, Device};

fn main() -> anyhow::Result<()> {
    let p = "/models/comfyui/text_encoders/qwen3vl_4b_fp8_scaled.safetensors";
    let archive = SafeTensorsArchive::open(p)?;
    let dev = Device::Cpu;

    for layer_idx in [0, 5, 10, 20, 35] {
        let k1 = format!("model.layers.{}.input_layernorm.weight", layer_idx);
        let k2 = format!("model.layers.{}.post_attention_layernorm.weight", layer_idx);
        for k in [&k1, &k2] {
            if let Ok(t) = archive.get_tensor(k, &dev, DType::F32) {
                let mean = t.mean_all()?.to_scalar::<f32>()?;
                let min = t.flatten_all()?.to_vec1::<f32>()?.into_iter().fold(f32::INFINITY, f32::min);
                let max = t.flatten_all()?.to_vec1::<f32>()?.into_iter().fold(f32::NEG_INFINITY, f32::max);
                println!("   • {:<50} mean={:+.4} | min={:+.4} | max={:+.4}", k, mean, min, max);
            }
        }
    }
    Ok(())
}
