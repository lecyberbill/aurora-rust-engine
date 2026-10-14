fn main() {
    // Ensure Windows MSVC linker finds stdc++.lib stub for candle-flash-attn
    println!("cargo:rustc-link-search=native=.");

    #[cfg(feature = "rocm")]
    {
        let out_dir = std::env::var("OUT_DIR").unwrap();
        let kernels = [
            ("src/kernels/wmma_q4k_rdna4.hip", "wmma_q4k_rdna4"),
            ("src/kernels/fused_sdpa_rdna4.hip", "fused_sdpa_rdna4"),
        ];

        let mut compiled_objs = Vec::new();

        for (source, name) in kernels {
            if std::path::Path::new(source).exists() {
                println!("cargo:rerun-if-changed={source}");
                let out_obj = format!("{out_dir}/{name}.o");
                let status = std::process::Command::new("/opt/rocm/bin/hipcc")
                    .args([
                        "-c",
                        source,
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
                        compiled_objs.push(out_obj);
                    }
                }
            }
        }

        if !compiled_objs.is_empty() {
            let out_lib = format!("{out_dir}/libaurora_rocm_kernels.a");
            let mut ar_args = vec!["rcs".to_string(), out_lib];
            ar_args.extend(compiled_objs);
            let ar_status = std::process::Command::new("ar")
                .args(&ar_args)
                .status();
            if let Ok(ar_st) = ar_status {
                if ar_st.success() {
                    println!("cargo:rustc-link-search=native={out_dir}");
                    println!("cargo:rustc-link-lib=static=aurora_rocm_kernels");
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

