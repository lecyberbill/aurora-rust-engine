// [WFGY] Zone: SAFE | λ: 0.15 | Fallbacks: 0 | Action: Inspect tensor keys in fluxKleinFP8_flux2Klein4bFp8.safetensors

use safetensors::SafeTensors;
use memmap2::Mmap;
use std::fs::File;
use std::collections::BTreeSet;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let klein_path = if args.len() > 1 {
        args[1].clone()
    } else {
        "G:\\models\\flux\\flux2Klein_4b.safetensors".to_string()
    };
    println!("================================================================================");
    println!("🔍 Inspecting Flux.2 Klein 4B FP8: {}", klein_path);
    println!("================================================================================");

    let file = File::open(klein_path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;

    let mut prefixes = BTreeSet::new();
    let mut total_params = 0usize;
    let mut double_block_count = 0;
    let mut single_block_count = 0;

    for (name, view) in st.tensors() {
        total_params += view.shape().iter().product::<usize>();
        if let Some(prefix) = name.split('.').next() {
            prefixes.insert(prefix.to_string());
        }
        if name.starts_with("double_blocks.") && name.ends_with(".img_mod.lin.weight") {
            double_block_count += 1;
        }
        if name.starts_with("single_blocks.") && name.ends_with(".modulation.lin.weight") {
            single_block_count += 1;
        }
    }

    println!("📊 Total Tensors: {} | Total Parameters: {:.2} Billion", st.tensors().len(), total_params as f64 / 1e9);
    println!("📁 Root Prefixes: {:?}", prefixes);

    let mut keys: Vec<String> = st.tensors().iter().map(|(k, _)| k.to_string()).collect();
    keys.sort();
    println!("\n📋 Complete Tensor Names List ({} tensors):", keys.len());
    for k in &keys {
        let view = st.tensor(k)?;
        println!("   • {:<60} {:?}", k, view.shape());
    }

    Ok(())
}
