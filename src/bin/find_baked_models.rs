// [WFGY] Zone: SAFE | λ: 0.15 | Fallbacks: 0 | Action: Check all models in G:\models\flux for embedded VAE

use safetensors::SafeTensors;
use memmap2::Mmap;
use std::fs::{self, File};
use std::path::Path;

fn check_file(path: &Path) {
    if let Ok(file) = File::open(path) {
        if let Ok(mmap) = unsafe { Mmap::map(&file) } {
            if let Ok(st) = SafeTensors::deserialize(&mmap) {
                let mut vae_count = 0;
                let mut text_count = 0;
                for (name, _) in st.tensors() {
                    if name.starts_with("vae.") || name.starts_with("first_stage_model.") || name.starts_with("decoder.") {
                        vae_count += 1;
                    }
                    if name.starts_with("text_encoders.") || name.starts_with("t5xxl.") || name.starts_with("clip_l.") {
                        text_count += 1;
                    }
                }
                if vae_count > 0 || text_count > 0 {
                    println!("✨ [ALL-IN-ONE CHECKPOINT] {}: {} VAE tensors, {} Text tensors", path.file_name().unwrap_or_default().to_string_lossy(), vae_count, text_count);
                }
            }
        }
    }
}

fn main() {
    println!("🔍 Checking for All-in-One Checkpoints with Baked VAE and Text Encoders...");
    if let Ok(entries) = fs::read_dir("G:\\models\\flux") {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.extension().map(|e| e == "safetensors").unwrap_or(false) {
                check_file(&p);
            }
        }
    }
    if let Ok(entries) = fs::read_dir("G:\\models\\SD3") {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.extension().map(|e| e == "safetensors").unwrap_or(false) {
                check_file(&p);
            }
        }
    }
}
