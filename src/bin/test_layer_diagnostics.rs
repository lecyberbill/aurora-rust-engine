use anyhow::{Context, Result};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::VarBuilder;
use std::path::PathBuf;

use aurora_rust_engine::diffusion::dit::cogvideox::{CogVideoXConfig, CogVideoXTransformer3D};

fn main() -> Result<()> {
    println!("============================================================");
    println!("  CogVideoX-2B Layer-by-Layer Numerical Diagnostic Tool");
    println!("============================================================");

    let dev = match aurora_rust_engine::device::auto_device() {
        Ok(d) => d,
        Err(_) => Device::Cpu,
    };
    let dtype = if dev.is_cpu() { DType::F32 } else { DType::F16 };
    println!("[*] Device: {:?}, Precision: {:?}", dev, dtype);

    let gt_path = PathBuf::from("/models/outputs/intermediate_groundtruth.safetensors");
    let in_path = PathBuf::from("/models/outputs/test_groundtruth.safetensors");

    let gt_tensors = candle_core::safetensors::load(&gt_path, &dev)?;
    let in_tensors = candle_core::safetensors::load(&in_path, &dev)?;

    let hidden_states = in_tensors.get("hidden_states").cloned().context("hidden_states")?.to_dtype(dtype)?;
    let encoder_hidden_states = in_tensors.get("encoder_hidden_states").cloned().context("encoder_hidden_states")?.to_dtype(dtype)?;
    let timestep = in_tensors.get("timestep").cloned().context("timestep")?.to_dtype(dtype)?;

    let emb_gt = gt_tensors.get("emb").cloned().context("emb")?.to_dtype(dtype)?;
    let patch_embed_gt = gt_tensors.get("patch_embed_out").cloned().context("patch_embed_out")?.to_dtype(dtype)?;

    // Load model
    let weights_path = PathBuf::from("/models/comfyui/diffusion_models/cogvideox_2b_transformer.safetensors");
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&weights_path], dtype, &dev)? };
    let config = CogVideoXConfig::default();
    let mut model = CogVideoXTransformer3D::new(config, vb)?;

    let pos_path = PathBuf::from("/models/outputs/cogvideox_pos_embed.safetensors");
    if pos_path.exists() {
        let pos_tensors = candle_core::safetensors::load(&pos_path, &dev)?;
        if let Some(pos) = pos_tensors.get("pos_embedding") {
            model.patch_embed.set_pos_embedding(pos.to_dtype(dtype)?.to_device(&dev)?);
        }
    }

    // 1. Time embedding diagnostic
    println!("\n--- [Step 1: Time Embedding] ---");
    let temb = model.time_embed.forward(&timestep)?;
    let d_temb = (&temb.to_dtype(DType::F32)? - &emb_gt.to_dtype(DType::F32)?)?.abs()?;
    let max_d_temb = d_temb.flatten_all()?.max(0)?.to_scalar::<f32>()?;
    println!("Time Embedding Max Diff: {:.6e}", max_d_temb);

    // 2. Patch embedding diagnostic
    println!("\n--- [Step 2: Patch Embedding] ---");
    let rust_input = hidden_states.transpose(1, 2)?; // [B, 16, T, H, W]
    let video_tokens = model.patch_embed.forward_video(&rust_input)?;
    let text_tokens = model.patch_embed.forward_text(&encoder_hidden_states)?;

    let (b, _, t, h, w) = rust_input.dims5()?;
    let p = model.cfg.patch_size;
    let h_p = h / p;
    let w_p = w / p;

    let embeds = Tensor::cat(&[&text_tokens, &video_tokens], 1)?;
    let total_seq = embeds.dim(1)?;
    let text_seq_len = text_tokens.dim(1)?;
    let pos_emb = model.patch_embed.pos_embedding.as_ref().unwrap();
    let embeds = (&embeds + pos_emb)?;

    let d_patch = (&embeds.to_dtype(DType::F32)? - &patch_embed_gt.to_dtype(DType::F32)?)?.abs()?;
    let max_d_patch = d_patch.flatten_all()?.max(0)?.to_scalar::<f32>()?;
    let mean_d_patch = d_patch.flatten_all()?.mean(0)?.to_scalar::<f32>()?;
    println!("Patch Embedding Max Diff : {:.6e}", max_d_patch);
    println!("Patch Embedding Mean Diff: {:.6e}", mean_d_patch);

    let enc_in = embeds.narrow(1, 0, text_seq_len)?;
    let vid_in = embeds.narrow(1, text_seq_len, total_seq - text_seq_len)?;

    // 3. Loop through all 30 transformer blocks
    println!("\n--- [Step 3: Transformer Blocks Evolution] ---");
    let mut cur_vid = vid_in;
    let mut cur_enc = enc_in;

    for i in 0..model.blocks.len() {
        let (next_vid, next_enc) = model.blocks[i].forward(&cur_vid, &cur_enc, &temb, None)?;
        cur_vid = next_vid;
        cur_enc = next_enc;

        if i == 0 || i == 1 || i == 5 || i == 10 || i == 20 || i == 29 {
            let vid_f32 = cur_vid.to_dtype(DType::F32)?;
            let mean_val = vid_f32.flatten_all()?.mean(0)?.to_scalar::<f32>()?;
            let std_val = vid_f32.flatten_all()?.sqr()?.mean(0)?.to_scalar::<f32>()?.sqrt();
            println!("Block {:02} Vid Out: Mean = {:.4}, RMS = {:.4}", i, mean_val, std_val);
        }
    }

    // 4. Final AdaLN modulation & Output projection on video stream
    println!("\n--- [Step 4: Final Head & Output] ---");
    let hidden_norm = model.norm_final.forward(&cur_vid)?;
    let temb_act = candle_nn::ops::silu(&temb)?;
    let mod_out = model.norm_out_linear.forward(&temb_act)?;
    let mod_chunks = mod_out.chunk(2, 1)?;
    let shift_out = mod_chunks[0].unsqueeze(1)?;
    let scale_out = mod_chunks[1].unsqueeze(1)?;

    let x_out_norm = model.norm_out_norm.forward(&hidden_norm)?;
    let x_final = x_out_norm.broadcast_mul(&(scale_out + 1.0)?)?.broadcast_add(&shift_out)?;
    let projected = model.proj_out.forward(&x_final)?;

    let unpatch = projected
        .reshape((b * t, h_p, w_p, 16, p, p))?
        .permute((0, 3, 1, 4, 2, 5))?
        .contiguous()?
        .reshape((b * t, 16, h, w))?
        .reshape((b, t, 16, h, w))?;

    let gt_out = in_tensors.get("diffusers_output").unwrap().to_dtype(dtype)?;
    let d_final = (&unpatch.to_dtype(DType::F32)? - &gt_out.to_dtype(DType::F32)?)?.abs()?;
    let max_d_final = d_final.flatten_all()?.max(0)?.to_scalar::<f32>()?;
    let mean_d_final = d_final.flatten_all()?.mean(0)?.to_scalar::<f32>()?;

    let u_f32 = unpatch.to_dtype(DType::F32)?;
    let g_f32 = gt_out.to_dtype(DType::F32)?;
    let u_mean = u_f32.flatten_all()?.mean(0)?.to_scalar::<f32>()?;
    let g_mean = g_f32.flatten_all()?.mean(0)?.to_scalar::<f32>()?;

    println!("Diffusers Final Mean : {:.6}", g_mean);
    println!("Candle Final Mean    : {:.6}", u_mean);
    println!("Final Max Diff       : {:.6}", max_d_final);
    println!("Final Mean Diff      : {:.6}", mean_d_final);

    println!("\n============================================================");

    Ok(())
}
