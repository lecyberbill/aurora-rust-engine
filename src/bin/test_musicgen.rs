// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: MusicGen text-to-music end-to-end

use aurora_rust_engine::pipelines::MusicgenPipeline;

fn main() -> anyhow::Result<()> {
    let argv: Vec<String> = std::env::args().collect();
    let mut dir = "D:/models/musicgen-small".to_string();
    let mut prompt = "lo-fi hip hop beat, mellow piano, vinyl crackle, 80 bpm".to_string();
    let mut frames = 250usize; // 5 s at 50 Hz
    let mut guidance = 3.0f32;
    let mut seed = 0u64;
    let mut out = "outputs/audio_showcase/musicgen_out.wav".to_string();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "-m" | "--model-dir" => { dir = argv[i + 1].clone(); i += 1; }
            "-p" | "--prompt" => { prompt = argv[i + 1].clone(); i += 1; }
            "-f" | "--frames" => { frames = argv[i + 1].parse().unwrap_or(frames); i += 1; }
            "-g" | "--guidance" => { guidance = argv[i + 1].parse().unwrap_or(guidance); i += 1; }
            "--seed" => { seed = argv[i + 1].parse().unwrap_or(seed); i += 1; }
            "-o" | "--out" => { out = argv[i + 1].clone(); i += 1; }
            _ => {}
        }
        i += 1;
    }
    let t0 = std::time::Instant::now();
    let pipe = MusicgenPipeline::from_pretrained(&dir)?;
    println!("loaded ({:?}) in {:.1}s", pipe.device, t0.elapsed().as_secs_f32());
    let audio = pipe.generate(&prompt, frames, guidance, 1.0, 50, seed)?;
    if let Some(d) = std::path::Path::new(&out).parent() { std::fs::create_dir_all(d)?; }
    audio.save_auto(&out)?;
    println!("saved {out} ({:.2}s, {:.1}s total)", audio.duration_seconds(), t0.elapsed().as_secs_f32());
    Ok(())
}
