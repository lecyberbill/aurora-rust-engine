// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: Stable Audio Open text-to-audio end-to-end

use aurora_rust_engine::pipelines::StableAudioPipeline;

fn main() -> anyhow::Result<()> {
    let argv: Vec<String> = std::env::args().collect();
    let mut model_dir = "G:/models/Audio/stable-audio-open-models".to_string();
    let mut prompt = "The sound of a hammer hitting a wooden surface.".to_string();
    let mut negative = "Low quality.".to_string();
    let mut end = 10.0f32;
    let mut steps = 100usize;
    let mut guidance = 7.0f32;
    let mut seed = 0u64;
    let mut out = "outputs/audio_showcase/sao_out.wav".to_string();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "-m" | "--model-dir" => { model_dir = argv[i + 1].clone(); i += 1; }
            "-p" | "--prompt" => { prompt = argv[i + 1].clone(); i += 1; }
            "--negative" => { negative = argv[i + 1].clone(); i += 1; }
            "-d" | "--duration" => { end = argv[i + 1].parse().unwrap_or(end); i += 1; }
            "-s" | "--steps" => { steps = argv[i + 1].parse().unwrap_or(steps); i += 1; }
            "-g" | "--guidance" => { guidance = argv[i + 1].parse().unwrap_or(guidance); i += 1; }
            "--seed" => { seed = argv[i + 1].parse().unwrap_or(seed); i += 1; }
            "-o" | "--out" => { out = argv[i + 1].clone(); i += 1; }
            _ => {}
        }
        i += 1;
    }
    let t0 = std::time::Instant::now();
    let pipe = StableAudioPipeline::from_pretrained(&model_dir)?;
    println!("pipeline loaded ({:?}) in {:.1}s", pipe.device, t0.elapsed().as_secs_f32());
    let audio = pipe.generate(&prompt, &negative, 0.0, end, steps, guidance, seed)?;
    if let Some(dir) = std::path::Path::new(&out).parent() { std::fs::create_dir_all(dir)?; }
    audio.save_auto(&out)?;
    println!("saved {out} ({:.2}s, {:.1}s total)", audio.duration_seconds(), t0.elapsed().as_secs_f32());
    Ok(())
}
