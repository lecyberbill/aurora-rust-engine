use std::env;
use aurora_rust_engine::weights;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: inspect_archive <path_to_weights>");
        std::process::exit(1);
    }
    let path = &args[1];
    let src = weights::open_auto(path)?;
    println!("Total keys in {}: {}", path, src.keys().len());
    let mut keys: Vec<String> = src.keys().to_vec();
    keys.sort();
    for k in keys {
        if let Some((dtype, shape)) = src.raw_info(&k) {
            println!("  {:55} | {:?} | {:?}", k, dtype, shape);
        }
    }
    Ok(())
}
