use std::process::{Command, exit};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let status = Command::new(r#"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\nvcc.exe"#)
        .arg("-DCCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING")
        .args(&args)
        .status();

    match status {
        Ok(s) => exit(s.code().unwrap_or(1)),
        Err(e) => {
            eprintln!("nvcc_wrapper failed: {e}");
            exit(1);
        }
    }
}
