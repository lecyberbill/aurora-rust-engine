// [WFGY] Zone: SAFE | λ: 0.15 | Fallbacks: 0 | Action: Dump all keys containing conv_shortcut or nin_shortcut in decoder

use safetensors::SafeTensors;
use memmap2::Mmap;
use std::fs::File;

fn main() -> anyhow::Result<()> {
    let flux_path = "G:\\models\\flux\\flux1SchnellFp8_schnellFp8.safetensors";
    let file = File::open(flux_path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;

    println!("🔍 Searching for shortcut keys in decoder:");
    for (name, view) in st.tensors() {
        if name.contains("decoder.") && (name.contains("shortcut") || name.contains("nin")) {
            println!("   • {:<60} {:?} {:?}", name, view.dtype(), view.shape());
        }
    }

    Ok(())
}
