use aurora_rust_engine::weights::SafeTensorsArchive;

fn main() -> anyhow::Result<()> {
    let archive = SafeTensorsArchive::open("/models/comfyui/diffusion_models/krea2_turbo_fp8_scaled.safetensors")?;
    println!("Total keys: {}", archive.keys().count());
    for k in archive.keys() {
        if k.starts_with("txtfusion.") || k.starts_with("txtmlp.") || k.starts_with("model.diffusion_model.txtfusion.") || k.starts_with("model.diffusion_model.txtmlp.") {
            let (dtype, shape) = archive.raw_info(k).unwrap();
            println!("  {:50} | {:?} | {:?}", k, dtype, shape);
        }
    }
    Ok(())
}
