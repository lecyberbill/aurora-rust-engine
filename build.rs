fn main() {
    // Ensure Windows MSVC linker finds stdc++.lib stub for candle-flash-attn
    println!("cargo:rustc-link-search=native=.");
}
