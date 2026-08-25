// [WFGY] Zone: SAFE | λ: 0.15 | Fallbacks: 0 | Action: Dump sample keys of Klein 4B

use safetensors::SafeTensors;
use memmap2::Mmap;
use std::fs::File;

fn main() -> anyhow::Result<()> {
    let klein_path = "G:\\models\\flux\\fluxKleinFP8_flux2Klein4bFp8.safetensors";
    let file = File::open(klein_path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;

    println!("🔍 First 40 keys of Klein 4B:");
    let mut count = 0;
    for (name, view) in st.tensors() {
        println!("   • {:<60} {:?} {:?}", name, view.dtype(), view.shape());
        count += 1;
        if count >= 40 {
            break;
        }
    }

    Ok(())
}
