// [WFGY] Zone: TRANSIT | λ: 0.15 | Fallbacks: 0 | Action: Pure Rust Multimodal Gemma 4 Vision CLI on ROCm
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor, quantized::gguf_file};
use candle_nn::VarBuilder;
use image::ImageReader;
use tokenizers::Tokenizer;

use aurora_rust_engine::models::quantized_gemma3::ModelWeights;
use aurora_rust_engine::models::vlm::Gemma4VisionTower;

fn main() -> Result<()> {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║   AURORA RUST ENGINE — GEMMA 4 31B MULTIMODAL VLM (ROCm)   ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    let image_path = PathBuf::from("/models/llm/Le-Complexe.jpg");
    let model_path = PathBuf::from("/models/llm/gemma-4-31B-it-UD-Q4_K_XL.gguf");
    let mmproj_path = PathBuf::from("/models/llm/mmproj-gemma-4-31b-BF16.gguf");
    let tok_path = "gemma4_tokenizer.json";

    #[cfg(feature = "rocm")]
    let dev = Device::new_rocm(0)?;
    #[cfg(not(feature = "rocm"))]
    let dev = Device::Cpu;

    println!("⚡ Compute Device: {:?}", dev);
    println!("🖼️  Image Path:     {}", image_path.display());
    println!("📦 LLM GGUF:       {}", model_path.display());
    println!("👁️  Vision mmproj:  {}", mmproj_path.display());

    // 1. Load image and display dimensions
    let t0 = Instant::now();
    let img = ImageReader::open(&image_path)?.decode()?;
    println!("📸 Image loaded: {}x{} in {:.2}ms", img.width(), img.height(), t0.elapsed().as_secs_f32() * 1000.0);

    // 2. Load LLM Weights
    println!("\nLoading Gemma 4 31B LLM weights...");
    let t_load = Instant::now();
    let mut file = File::open(&model_path)?;
    let content = gguf_file::Content::read(&mut file)?;
    let mut model = ModelWeights::from_gguf(content, &mut file, &dev)?;
    println!("✅ LLM loaded in {:.2}s", t_load.elapsed().as_secs_f32());

    // 3. Load Vision Tower & Projector from mmproj GGUF
    println!("\nLoading Vision Tower & Projector from mmproj GGUF...");
    let t_vload = Instant::now();
    let mut mm_file = File::open(&mmproj_path)?;
    let mm_content = gguf_file::Content::read(&mut mm_file)?;
    println!("✅ mmproj opened: {} tensors found", mm_content.tensor_infos.len());

    let mut tensors = std::collections::HashMap::new();
    for (name, _) in &mm_content.tensor_infos {
        let t = mm_content.tensor(&mut mm_file, name, &dev)?.dequantize(&dev)?;
        tensors.insert(name.clone(), t);
    }
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &dev);
    let vision_tower = Gemma4VisionTower::new(vb)?;
    println!("✅ Gemma 4 Vision Tower loaded in {:.2}s", t_vload.elapsed().as_secs_f32());

    // 4. Tokenizer & Prompt with multimodal tags
    let tokenizer = Tokenizer::from_file(tok_path).map_err(|e| anyhow::anyhow!(e))?;

    // Compute aspect-ratio preserving size for Le-Complexe.jpg:
    // 500x679 -> target_w=672, target_h=912 -> 42x57 patches = 2394 patches -> 266 soft tokens
    let orig_w = img.width() as f64;
    let orig_h = img.height() as f64;
    let patch_size = 16.0;
    let pooling_k = 3.0;
    let side_mult = patch_size * pooling_k; // 48.0
    let max_patches = 280.0 * 9.0; // 2520.0
    let factor = (max_patches * patch_size * patch_size / (orig_w * orig_h)).sqrt();
    let target_w = ((orig_w * factor / side_mult).floor() * side_mult) as u32;
    let target_h = ((orig_h * factor / side_mult).floor() * side_mult) as u32;
    println!("📐 Aspect-ratio preserving resize: {}x{} -> {}x{}", img.width(), img.height(), target_w, target_h);

    let num_soft_tokens = (target_w / 48) * (target_h / 48); // 14 * 19 = 266
    println!("🔢 Number of visual soft tokens: {}", num_soft_tokens);

    let resized = img.resize_exact(target_w, target_h, image::imageops::FilterType::CatmullRom);
    let rgb = resized.to_rgb8();
    let raw = rgb.into_raw();
    let num_pixels = (target_w * target_h) as usize;
    let mut float_data = vec![0f32; 3 * num_pixels];
    for i in 0..num_pixels {
        float_data[i] = raw[i * 3] as f32 / 255.0;
        float_data[num_pixels + i] = raw[i * 3 + 1] as f32 / 255.0;
        float_data[2 * num_pixels + i] = raw[i * 3 + 2] as f32 / 255.0;
    }
    let pixel_tensor = Tensor::from_vec(float_data, (1, 3, target_h as usize, target_w as usize), &dev)?;
    println!("🖼️  Pixel Tensor shape: {:?}", pixel_tensor.shape());

    // Run Vision Tower forward pass: [1, 3, H, W] -> [1, 266, 5376]
    println!("\nComputing image embeddings via Gemma4VisionTower...");
    let t_vis = Instant::now();
    let visual_embeds = vision_tower.forward(&pixel_tensor)?;
    println!("✅ Visual embeddings computed: {:?} in {:.2}ms", visual_embeds.shape(), t_vis.elapsed().as_secs_f32() * 1000.0);

    // Build Multimodal Prompt
    // Format: <|turn>user\n<|image><|image|>...<|image|><image|>\nDécris en détail cette image et lis les textes visibles.<turn|>\n<|turn>model\n<|channel>thought\n<channel|>
    let img_placeholders = "<|image|>".repeat(num_soft_tokens as usize);
    let prompt = format!(
        "<|turn>user\n<|image>{img_placeholders}<image|>\nDécris en détail cette image et lis tous les textes visibles.<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
    );

    let mut tokens = tokenizer.encode(prompt.as_str(), false).map_err(|e| anyhow::anyhow!(e))?.get_ids().to_vec();
    if tokens.is_empty() || tokens[0] != 2 {
        tokens.insert(0, 2); // Prepend BOS
    }

    println!("\n=== Multimodal Instruct Prompt ===");
    println!("Prompt tokens total: {}", tokens.len());

    // Compute text embeddings
    let tok_tensor = Tensor::new(tokens.as_slice(), &dev)?.unsqueeze(0)?;
    let text_embeds = model.embed_tokens(&tok_tensor)?;
    println!("📝 Text embeddings shape: {:?}", text_embeds.shape());

    // Replace <|image|> token slots (ID 258880) with visual_embeds
    let image_token_id = 258880u32;
    let mut img_slot_indices = Vec::new();
    for (idx, &t) in tokens.iter().enumerate() {
        if t == image_token_id {
            img_slot_indices.push(idx);
        }
    }
    println!("🎯 Found {} <|image|> slots in prompt", img_slot_indices.len());
    anyhow::ensure!(
        img_slot_indices.len() == num_soft_tokens as usize,
        "Mismatch: expected {} image slots, found {}",
        num_soft_tokens,
        img_slot_indices.len()
    );

    let start_slot = img_slot_indices[0];
    let end_slot = img_slot_indices[img_slot_indices.len() - 1] + 1;
    let prefix_embeds = text_embeds.i((.., ..start_slot, ..))?;
    let suffix_embeds = text_embeds.i((.., end_slot.., ..))?;

    let merged_embeds = Tensor::cat(&[&prefix_embeds, &visual_embeds, &suffix_embeds], 1)?;
    println!("✨ Merged multimodal embeddings shape: {:?}", merged_embeds.shape());

    println!("\n=== Inférence en cours (Streaming ROCm) ===");
    std::io::stdout().flush()?;

    let max_gen_tokens = 256;
    let start_gen = Instant::now();
    let mut gen_count = 0;

    // Prefill pass with merged multimodal embeddings
    let initial_len = merged_embeds.dim(1)?;
    let logits = model.forward_embeds(merged_embeds, 0)?;
    let last_logits: Vec<f32> = logits.to_dtype(DType::F32)?.squeeze(0)?.to_vec1()?;

    let mut max_id = 0;
    let mut max_logit = f32::NEG_INFINITY;
    for (id, val) in last_logits.into_iter().enumerate() {
        if val > max_logit {
            max_logit = val;
            max_id = id;
        }
    }

    tokens.push(max_id as u32);
    gen_count += 1;
    let piece = tokenizer.decode(&[max_id as u32], false).unwrap_or_default();
    print!("{}", piece);
    std::io::stdout().flush()?;

    // Autoregressive token decoding
    for pos in initial_len..(initial_len + max_gen_tokens) {
        let step_tok = Tensor::new(&[tokens[tokens.len() - 1]], &dev)?.unsqueeze(0)?;
        let logits = model.forward(&step_tok, pos)?;
        let last_logits: Vec<f32> = logits.to_dtype(DType::F32)?.squeeze(0)?.to_vec1()?;

        let mut max_id = 0;
        let mut max_logit = f32::NEG_INFINITY;
        for (id, val) in last_logits.into_iter().enumerate() {
            if val > max_logit {
                max_logit = val;
                max_id = id;
            }
        }

        tokens.push(max_id as u32);
        gen_count += 1;

        let piece = tokenizer.decode(&[max_id as u32], false).unwrap_or_default();
        print!("{}", piece);
        std::io::stdout().flush()?;

        if max_id == 1 || max_id == 106 || piece.contains("<turn|>") {
            break;
        }
    }

    let elapsed = start_gen.elapsed();
    println!("\n\n--- Inférence terminée : {} tokens en {:.2?} ({:.2} tok/s) ---", gen_count, elapsed, gen_count as f64 / elapsed.as_secs_f64());

    Ok(())
}
