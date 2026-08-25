// [WFGY] Zone: SAFE | λ: 0.15 | Fallbacks: 0 | Action: Dump time_in, vector_in, and double_blocks.0 tensor keys

use safetensors::SafeTensors;
use memmap2::Mmap;
use std::fs::File;

fn main() -> anyhow::Result<()> {
    let flux_path = "G:\\models\\flux\\flux1_v10Fp8Schnell.safetensors";
    let file = File::open(flux_path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;

    println!("🔍 Dumping time_in, vector_in, img_in, and double_blocks.0 keys:");
    for (name, view) in st.tensors() {
        if name.starts_with("time_in") || name.starts_with("vector_in") || name.starts_with("guidance_in") 
           || name.starts_with("img_in") || name.starts_with("double_blocks.0") || name.starts_with("final_layer") {
            println!("   • {:<55} {:?} {:?}", name, view.dtype(), view.shape());
        }
    }

    Ok(())
}
