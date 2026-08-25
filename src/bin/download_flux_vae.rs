// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: Download official Flux.1 AutoEncoder VAE (ae.safetensors) via hf-hub

use hf_hub::api::sync::Api;
use std::fs;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    println!("================================================================================");
    println!("📥 DOWNLOADING OFFICIAL FLUX.1 VAE (ae.safetensors - ~319 MB)...");
    println!("================================================================================\n");

    let target_dir = Path::new("G:\\models\\vae");
    let target_file = target_dir.join("flux_ae.safetensors");

    if target_file.exists() {
        println!("✅ Flux VAE already exists at: {}", target_file.display());
        return Ok(());
    }

    println!("🌐 Connecting to Hugging Face Hub (Comfy-Org/Flux_pruned)...");
    let api = Api::new()?;
    let repo = api.model("Comfy-Org/Flux_pruned".to_string());
    let downloaded_path = repo.get("split_files/vae/ae.safetensors")?;

    println!("📦 Download complete at cache: {}", downloaded_path.display());
    fs::copy(&downloaded_path, &target_file)?;
    println!("🎉 Copied to permanent storage: {}", target_file.display());

    Ok(())
}
