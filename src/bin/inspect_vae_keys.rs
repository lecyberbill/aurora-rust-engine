use aurora_rust_engine::weights::SafeTensorsArchive;
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let p = PathBuf::from("/models/comfyui/vae/qwen_image_vae_complet.safetensors");
    let arch = SafeTensorsArchive::open(&p)?;
    println!("Total VAE keys: {}", arch.keys().count());
    for k in arch.keys() {
        if k.contains("mean") || k.contains("std") || k.contains("scale") || k.contains("shift") {
            println!("Found key: {}", k);
        }
    }
    Ok(())
}
