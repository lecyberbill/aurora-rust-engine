// [WFGY] Zone: SAFE | λ: 0.15 | Fallbacks: 0 | Action: Scan all safetensors files on all drives for Flux VAE and T5

use std::fs;
use std::path::Path;

fn search_flux_components(dir_path: &str) {
    let p = Path::new(dir_path);
    if !p.exists() { return; }

    if let Ok(entries) = fs::read_dir(p) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                let name = path.file_name().unwrap_or_default().to_string_lossy();
                if name.contains("ae.safetensors") || name.contains("flux_vae") || name.contains("t5") || name.contains("clip_l") {
                    let size_mb = entry.metadata().map(|m| m.len() as f64 / (1024.0 * 1024.0)).unwrap_or(0.0);
                    println!("   • [FOUND] {:<60} ({:.2} MB)", path.display(), size_mb);
                }
            } else if path.is_dir() {
                search_flux_components(&path.to_string_lossy());
            }
        }
    }
}

fn main() {
    println!("================================================================================");
    println!("🔍 Searching for Flux VAE (ae.safetensors) and T5 Encoders on disk...");
    println!("================================================================================\n");

    search_flux_components("G:\\models");
    search_flux_components("D:\\image_to_text");
    println!("\nSearch complete.");
}
