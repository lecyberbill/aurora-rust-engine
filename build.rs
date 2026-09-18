fn main() {
    // Ensure Windows MSVC linker finds stdc++.lib stub for candle-flash-attn
    println!("cargo:rustc-link-search=native=.");

    // Ensure MSVC cl.exe is in PATH for nvcc / candle-kernels compilation on Windows
    #[cfg(target_os = "windows")]
    {
        if let Ok(vswhere) = std::process::Command::new("C:\\Program Files (x86)\\Microsoft Visual Studio\\Installer\\vswhere.exe")
            .args(["-latest", "-products", "*", "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"])
            .output()
        {
            if vswhere.status.success() {
                let vs_path = String::from_utf8_lossy(&vswhere.stdout).trim().to_string();
                let msvc_dir = std::path::Path::new(&vs_path).join("VC").join("Tools").join("MSVC");
                if let Ok(entries) = std::fs::read_dir(msvc_dir) {
                    let mut versions: Vec<_> = entries.filter_map(|e| e.ok()).map(|e| e.path()).collect();
                    versions.sort();
                    if let Some(latest) = versions.last() {
                        let cl_dir = latest.join("bin").join("Hostx64").join("x64");
                        if cl_dir.exists() {
                            if let Ok(curr_path) = std::env::var("PATH") {
                                std::env::set_var("PATH", format!("{};{}", cl_dir.display(), curr_path));
                            }
                        }
                    }
                }
            }
        }
    }
}

