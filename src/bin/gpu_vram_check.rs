// [WFGY] Zone: SAFE | λ: 0.15 | Fallbacks: 0 | Action: Check GPU VRAM and process consumption via nvidia-smi

use std::process::Command;

fn main() {
    println!("================================================================================");
    println!("🔍 NVIDIA GPU VRAM & Process Diagnostic");
    println!("================================================================================\n");

    let output = Command::new("nvidia-smi")
        .output();

    match output {
        Ok(out) => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            println!("{}", stdout);
        }
        Err(e) => {
            eprintln!("[-] Failed to execute nvidia-smi: {}", e);
        }
    }
}
