// [WFGY] Zone: SAFE | λ: 0.15 | Fallbacks: 0 | Action: Dump all VAE tensor keys from flux1SchnellFp8_schnellFp8.safetensors

use safetensors::SafeTensors;
use memmap2::Mmap;
use std::fs::File;

fn main() -> anyhow::Result<()> {
    let flux_path = "G:\\models\\flux\\flux1SchnellFp8_schnellFp8.safetensors";
    let file = File::open(flux_path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;

    println!("🔍 Dumping all VAE tensor keys:");
    let mut count = 0;
    for (name, view) in st.tensors() {
        if name.starts_with("vae.") || name.starts_with("first_stage_model.") {
            println!("   • {:<60} {:?} {:?}", name, view.dtype(), view.shape());
            count += 1;
            if count >= 30 {
                break;
            }
        }
    }

    Ok(())
}
