// [WFGY] Zone: SAFE | λ: 0.15 | Fallbacks: 0 | Action: Scan local model directory for Flux and SD3 weights

use std::fs;
use std::path::Path;

fn scan_dir(dir_path: &str) {
    println!("🔍 Scanning: {}", dir_path);
    let p = Path::new(dir_path);
    if !p.exists() {
        println!("   [-] Directory does not exist: {}", dir_path);
        return;
    }

    if let Ok(entries) = fs::read_dir(p) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                let size_mb = entry.metadata().map(|m| m.len() as f64 / (1024.0 * 1024.0)).unwrap_or(0.0);
                println!("   • [FILE] {:<50} ({:.2} MB / {:.2} GB)", path.file_name().unwrap_or_default().to_string_lossy(), size_mb, size_mb / 1024.0);
            } else if path.is_dir() {
                println!("   📁 [DIR]  {}", path.file_name().unwrap_or_default().to_string_lossy());
            }
        }
    }
}

fn main() {
    println!("================================================================================");
    println!("📂 Aurora Model Scanner for Flux.1 and SD3");
    println!("================================================================================\n");

    scan_dir("G:\\models\\flux");
    println!();
    scan_dir("G:\\models\\SD3");
    println!();
    scan_dir("G:\\models\\checkpoints");
}
