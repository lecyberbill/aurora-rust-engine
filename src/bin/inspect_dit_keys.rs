// [WFGY] Zone: SAFE | λ: 0.15 | Fallbacks: 0 | Action: Inspect tensor keys in Flux and SD3 checkpoints

use safetensors::SafeTensors;
use memmap2::Mmap;
use std::fs::File;
use std::collections::BTreeSet;

fn inspect_file(path: &str) -> anyhow::Result<()> {
    println!("================================================================================");
    println!("🔍 Inspecting Checkpoint: {}", path);
    println!("================================================================================");

    let file = File::open(path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;

    let mut prefixes = BTreeSet::new();
    let mut sample_keys = Vec::new();
    let mut total_params = 0usize;

    for (name, view) in st.tensors() {
        let num_elements = view.shape().iter().product::<usize>();
        total_params += num_elements;

        if let Some(prefix) = name.split('.').next() {
            prefixes.insert(prefix.to_string());
        }

        if sample_keys.len() < 30 {
            sample_keys.push((name.to_string(), view.dtype(), view.shape().to_vec()));
        }
    }

    println!("📊 Total Tensors: {} | Total Parameters: {:.2} Billion", st.tensors().len(), total_params as f64 / 1e9);
    println!("📁 Detected Root Prefixes: {:?}", prefixes);
    println!("\n🔍 First 30 Sample Tensor Keys:");
    for (name, dtype, shape) in sample_keys {
        println!("   • {:<60} {:?} {:?}", name, dtype, shape);
    }
    println!();
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let flux_path = "G:\\models\\flux\\flux1_v10Fp8Schnell.safetensors";
    let sd3_path = "G:\\models\\SD3\\stableDiffusion35Fp8_v35LargeTurbo.safetensors";

    if std::path::Path::new(flux_path).exists() {
        inspect_file(flux_path)?;
    }
    if std::path::Path::new(sd3_path).exists() {
        inspect_file(sd3_path)?;
    }

    Ok(())
}
