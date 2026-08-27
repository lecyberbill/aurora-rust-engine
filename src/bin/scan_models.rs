// [WFGY] Zone: SAFE | λ: 0.15 | Fallbacks: 0 | Action: Scan local model directory for Flux and SD3 weights

use std::fs;
use std::path::Path;

fn scan_recursive(p: &Path, depth: usize) {
    if depth > 4 { return; }
    if let Ok(entries) = fs::read_dir(p) {
        for entry in entries.flatten() {
            let path = entry.path();
            let name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
            if path.is_file() {
                let ext = path.extension().unwrap_or_default().to_string_lossy().to_lowercase();
                if ext == "safetensors" || ext == "json" || ext == "gguf" || ext == "bin" || ext == "pt" || name.contains("token") || name.contains("qwen") || name.contains("vae") {
                    let size_mb = entry.metadata().map(|m| m.len() as f64 / (1024.0 * 1024.0)).unwrap_or(0.0);
                    let indent = "  ".repeat(depth);
                    println!("{}• {:<60} ({:.2} MB)", indent, path.display(), size_mb);
                }
            } else if path.is_dir() {
                if !name.starts_with('.') && name != "target" && name != "node_modules" && name != "venv" && name != "__pycache__" {
                    let indent = "  ".repeat(depth);
                    println!("{}📁 {}", indent, path.display());
                    scan_recursive(&path, depth + 1);
                }
            }
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let target = if args.len() > 1 {
        args[1].clone()
    } else {
        "D:\\image_to_text\\Qpyt_image_gen".to_string()
    };
    println!("================================================================================");
    println!("📂 Aurora Deep Scanner: {}", target);
    println!("================================================================================\n");

    scan_recursive(Path::new(&target), 0);
}
