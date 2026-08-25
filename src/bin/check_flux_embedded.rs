// [WFGY] Zone: SAFE | λ: 0.15 | Fallbacks: 0 | Action: Check if VAE weights are included inside flux1_v10Fp8Schnell

use safetensors::SafeTensors;
use memmap2::Mmap;
use std::fs::File;

fn main() -> anyhow::Result<()> {
    let flux_path = "G:\\models\\flux\\flux1_v10Fp8Schnell.safetensors";
    let file = File::open(flux_path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;

    println!("🔍 Searching for VAE and Text Encoder tensors in Flux Checkpoint:");
    let mut vae_keys = Vec::new();
    let mut t5_keys = Vec::new();

    for (name, view) in st.tensors() {
        if name.contains("decoder.") || name.contains("encoder.") || name.contains("vae") || name.contains("first_stage_model") {
            vae_keys.push((name.to_string(), view.dtype(), view.shape().to_vec()));
        }
        if name.contains("t5") || name.contains("text_model") || name.contains("encoder.block") {
            t5_keys.push((name.to_string(), view.dtype(), view.shape().to_vec()));
        }
    }

    println!("📊 Found {} VAE tensors and {} T5/Text tensors", vae_keys.len(), t5_keys.len());
    for (name, dtype, shape) in vae_keys.iter().take(10) {
        println!("   • [VAE]  {:<50} {:?} {:?}", name, dtype, shape);
    }
    for (name, dtype, shape) in t5_keys.iter().take(10) {
        println!("   • [TEXT] {:<50} {:?} {:?}", name, dtype, shape);
    }

    Ok(())
}
