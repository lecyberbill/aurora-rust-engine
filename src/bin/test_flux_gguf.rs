use candle_core::{DType, Device, Result, Tensor};
use std::path::Path;
use std::time::Instant;
use aurora_rust_engine::pipelines::FluxPipeline;
use aurora_rust_engine::traits::DiffusionParams;

fn main() -> Result<()> {
    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    let path = "C:\\Users\\lecyb\\Downloads\\flux2-dev-Q8_0.gguf";
    let mistral_path = "G:\\models\\clip\\mistral_3_small_flux2_fp8.safetensors";
    let vae_path = "G:\\models\\vae\\flux2-vae.safetensors";

    if !Path::new(path).exists() {
        eprintln!("[-] GGUF not found: {}", path);
        return Ok(());
    }

    let mut pipeline = FluxPipeline::from_gguf(path, device.clone())
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    pipeline.enable_flash_attn();

    if Path::new(mistral_path).exists() {
        let mistral = aurora_rust_engine::text::Mistral3TextEncoder::from_safetensors(
            mistral_path, Some(std::path::Path::new("mistral_tokenizer.json")),
            Device::Cpu, DType::F16,
        ).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        pipeline.set_mistral(mistral);
        println!("✅ Mistral-3-Small Attached!");
    }
    if Path::new(vae_path).exists() {
        let va = aurora_rust_engine::weights::SafeTensorsArchive::open(vae_path)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let vr = aurora_rust_engine::weights::WeightRouter::new(&va, device.clone(), DType::F16);
        let vb = vr.vae_var_builder().map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        pipeline.set_vae(aurora_rust_engine::diffusion::vae_flux::FluxVaeDecoder::new(vb)?);
        println!("✅ Flux.2 VAE Decoder Attached!");
    }

    let prompt = "a gorgeous portrait of an arctic fox with sapphire blue eyes in a snowy forest at twilight, 8k";
    let steps: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(20);
    let guidance: f64 = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(3.5);

    let t0 = Instant::now();
    let params = DiffusionParams {
        prompt, negative_prompt: None, num_steps: steps, guidance_scale: guidance,
        width: 1024, height: 1024, seed: 42,
    };
    let (image, metrics) = pipeline.generate_with_metrics(params, None::<fn(usize,usize,&Tensor)>)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    let out = format!("outputs/flux_showcase/flux_dev_gguf_q8_s{}_g{}.png", steps, guidance);
    image.save(&out).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    println!("DONE {} steps, {}s, out={}", metrics.unet_steps, t0.elapsed().as_secs(), out);
    Ok(())
}
