fn main() {
    // Ensure Windows MSVC linker finds stdc++.lib stub for candle-flash-attn
    println!("cargo:rustc-link-search=native=.");

    #[cfg(feature = "rocm")]
    {
        let out_dir = std::env::var("OUT_DIR").unwrap();
        let hip_source = "src/kernels/wmma_q4k_rdna4.hip";
        let out_obj = format!("{out_dir}/wmma_q4k_rdna4.o");
        let out_lib = format!("{out_dir}/libwmma_q4k_rdna4.a");

        if std::path::Path::new(hip_source).exists() {
            println!("cargo:rerun-if-changed={hip_source}");
            let status = std::process::Command::new("/opt/rocm/bin/hipcc")
                .args([
                    "-c",
                    hip_source,
                    "-o",
                    &out_obj,
                    "--offload-arch=gfx1201",
                    "-O3",
                    "-fPIC",
                    "-D__HIP_PLATFORM_AMD__",
                ])
                .status();

            if let Ok(st) = status {
                if st.success() {
                    let ar_status = std::process::Command::new("ar")
                        .args(["rcs", &out_lib, &out_obj])
                        .status();
                    if let Ok(ar_st) = ar_status {
                        if ar_st.success() {
                            println!("cargo:rustc-link-search=native={out_dir}");
                            println!("cargo:rustc-link-lib=static=wmma_q4k_rdna4");
                        }
                    }
                }
            }
        }
    }

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

