// [WFGY] Zone: SAFE | λ: 0.15 | Fallbacks: 0 | Action: Dump shapes of decoder.up blocks

use safetensors::SafeTensors;
use memmap2::Mmap;
use std::fs::File;

fn main() -> anyhow::Result<()> {
    let flux_path = "G:\\models\\flux\\flux1SchnellFp8_schnellFp8.safetensors";
    let file = File::open(flux_path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;

    println!("🔍 Inspecting decoder.up tensor shapes:");
    for (name, view) in st.tensors() {
        if name.contains("decoder.up.") && name.contains("conv1.weight") {
            println!("   • {:<55} {:?} {:?}", name, view.dtype(), view.shape());
        }
    }

    Ok(())
}
