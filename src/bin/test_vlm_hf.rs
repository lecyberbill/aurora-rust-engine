// [WFGY] Zone: SAFE | λ: 0.25 | Fallbacks: 1 (HF weights) | Action: Pure Rust SmolVLM-500M-Instruct Visual Description Execution

use std::path::PathBuf;
use std::time::Instant;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use image::ImageReader;
use tokenizers::Tokenizer;
use aurora_rust_engine::hub::ModelHub;
use aurora_rust_engine::weights::SafeTensorsArchive;
use aurora_rust_engine::{
    select_device, VisionActivation, VisionTransformerConfig, VlmDecoderConfig, VlmModel
};

fn main() -> anyhow::Result<()> {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║       AURORA RUST ENGINE — REAL VLM VISUAL DESCRIPTION     ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    let repo_id = "HuggingFaceTB/SmolVLM-500M-Instruct";
    let image_path = PathBuf::from("outputs\\mem_probe.png");

    let device = select_device(true).unwrap_or(Device::Cpu);
    println!("⚡ Device: {:?}", device);

    let hub = ModelHub::from_env()?;
    let t0 = Instant::now();

    let weight_file = hub.resolve_hf(repo_id, "model.safetensors", None)?;
    let tokenizer_file = hub.resolve_hf(repo_id, "tokenizer.json", None)?;

    println!("📦 Weights:   {}", weight_file.display());
    println!("📦 Tokenizer: {}", tokenizer_file.display());

    let tokenizer = Tokenizer::from_file(&tokenizer_file).map_err(|e| anyhow::anyhow!("{e}"))?;

    let vision_cfg = VisionTransformerConfig {
        image_size: 512,
        patch_size: 16,
        num_channels: 3,
        embed_dim: 768,
        num_layers: 12,
        num_heads: 12,
        intermediate_size: 3072,
        spatial_merge_size: 4, // 4x4 merge = 16x compression -> 12288 dim
        act_type: VisionActivation::Gelu,
        layer_norm_eps: 1e-6,
    };

    let decoder_cfg = VlmDecoderConfig {
        vocab_size: 49280,
        hidden_size: 960,
        intermediate_size: 2560,
        num_hidden_layers: 32,
        num_attention_heads: 15,
        num_key_value_heads: 5,
        rms_norm_eps: 1e-5,
        rope_theta: 100_000.0,
        max_position_embeddings: 8192,
    };

    let image_token_id = 49190u32;

    let _archive = SafeTensorsArchive::open(&weight_file)?;

    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[weight_file.clone()], DType::F32, &device)?
    };

    let mut vlm = VlmModel::new(&vision_cfg, &decoder_cfg, image_token_id, vb)?;

    println!("✅ Model initialized on {:?} in {:.2}s", device, t0.elapsed().as_secs_f32());

    println!("🖼️  Opening image: {}...", image_path.display());
    let img = if image_path.exists() {
        ImageReader::open(&image_path)?.decode()?
    } else {
        image::DynamicImage::new_rgb8(512, 512)
    };
    println!("📸 Image Dimensions: {}x{}", img.width(), img.height());

    // Preprocessing: Resize to 512x512 normalized
    let resized = img.resize_exact(512, 512, image::imageops::FilterType::CatmullRom);
    let rgb = resized.to_rgb8();
    let raw = rgb.into_raw();
    let num_pixels = 512 * 512;
    let mut float_data = vec![0f32; 3 * num_pixels];
    for i in 0..num_pixels {
        float_data[i] = (raw[i * 3] as f32 / 255.0 - 0.5) / 0.5;
        float_data[num_pixels + i] = (raw[i * 3 + 1] as f32 / 255.0 - 0.5) / 0.5;
        float_data[2 * num_pixels + i] = (raw[i * 3 + 2] as f32 / 255.0 - 0.5) / 0.5;
    }
    let pixel_tensor = Tensor::from_vec(float_data, (1, 3, 512, 512), &device)?;

    let num_image_tokens = (512 / 16 / 4) * (512 / 16 / 4); // 8 * 8 = 64 visual tokens
    let prompt_text = "Describe in detail what you see in this image:";
    println!("\n💬 Prompt: \"{}\" (+ {} image tokens)", prompt_text, num_image_tokens);

    let encoding = tokenizer.encode(prompt_text, true).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut token_ids: Vec<u32> = vec![image_token_id; num_image_tokens];
    token_ids.extend_from_slice(encoding.get_ids());

    let input_ids = Tensor::new(&token_ids[..], &device)?.unsqueeze(0)?;

    println!("🚀 1. Multimodal Prefill (Image ViT + Token Embedding Splice)...");
    let t_prefill = Instant::now();
    let logits = vlm.forward(&input_ids, Some(&pixel_tensor), 0)?;
    println!("   -> Prefill completed in {:.2}ms", t_prefill.elapsed().as_secs_f32() * 1000.0);

    println!("🚀 2. Autoregressive Token Generation Loop...");
    let t_gen = Instant::now();
    let mut generated_tokens = Vec::new();
    let mut curr_pos = token_ids.len();

    // Pick top token from last prefill position
    let last_logits = logits.narrow(1, logits.dim(1)? - 1, 1)?.squeeze(0)?.squeeze(0)?;
    let mut next_token = last_logits.argmax(candle_core::D::Minus1)?.to_scalar::<u32>()?;

    for _ in 0..64 {
        if next_token == 2 || next_token == 0 || next_token == 1 { // EOS
            break;
        }
        generated_tokens.push(next_token);
        let next_input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
        let step_logits = vlm.forward(&next_input, None, curr_pos)?;
        curr_pos += 1;
        let step_squeezed = step_logits.squeeze(0)?.squeeze(0)?;
        next_token = step_squeezed.argmax(candle_core::D::Minus1)?.to_scalar::<u32>()?;
    }

    let gen_dur = t_gen.elapsed();
    let decoded_text = tokenizer.decode(&generated_tokens, true).unwrap_or_else(|_| "<decode error>".into());

    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║       📝 DESCRIPTION VISUELLE GÉNÉRÉE PAR SMOLVLM           ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!("{}\n", decoded_text);
    println!("⏱️  Temps de génération: {:.2}s (~{:.1} tok/s)", gen_dur.as_secs_f32(), generated_tokens.len() as f32 / gen_dur.as_secs_f32().max(0.001));

    Ok(())
}
