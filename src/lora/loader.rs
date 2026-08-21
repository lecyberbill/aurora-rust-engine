// [WFGY] Zone: SAFE | λ: 0.25 | Fallbacks: 1 (Format detection fallback) | Action: Multi-format LoRA weight parser (Kohya-ss, Diffusers, LyCORIS)

use candle_core::{DType, Device};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use crate::error::{LuminaError, Result};
use crate::weights::SafeTensorsArchive;
use super::types::{LoRAPair, LoRATarget, LoadedLoRA};

pub struct LoRALoader;

impl LoRALoader {
    pub fn load_from_file<P: AsRef<Path>>(
        path: P,
        multiplier: f64,
        _device: &Device,
        _dtype: DType,
    ) -> Result<LoadedLoRA> {
        let path_str = path.as_ref().to_string_lossy().to_string();
        let archive = SafeTensorsArchive::open(&path)?;
        let tensor_names = archive.tensor_names();

        // 1. Collect all down/up/alpha keys
        let mut base_names = HashSet::new();
        let mut alphas: HashMap<String, f64> = HashMap::new();

        for name in &tensor_names {
            if name.ends_with(".alpha") {
                let base = name.trim_end_matches(".alpha").to_string();
                if let Ok(alpha_t) = archive.get_tensor(name, &Device::Cpu, DType::F32) {
                    if let Ok(alpha_val) = alpha_t.to_vec0::<f32>() {
                        alphas.insert(base, alpha_val as f64);
                    }
                }
            } else if name.ends_with(".lora_down.weight") {
                let base = name.trim_end_matches(".lora_down.weight").to_string();
                base_names.insert(base);
            } else if name.ends_with(".lora_A.weight") {
                let base = name.trim_end_matches(".lora_A.weight").to_string();
                base_names.insert(base);
            } else if name.ends_with(".down.weight") {
                let base = name.trim_end_matches(".down.weight").to_string();
                base_names.insert(base);
            }
        }

        let mut pairs = Vec::new();

        for base in base_names {
            // Find down tensor
            let down_key = if tensor_names.contains(&format!("{}.lora_down.weight", base)) {
                format!("{}.lora_down.weight", base)
            } else if tensor_names.contains(&format!("{}.lora_A.weight", base)) {
                format!("{}.lora_A.weight", base)
            } else if tensor_names.contains(&format!("{}.down.weight", base)) {
                format!("{}.down.weight", base)
            } else {
                continue;
            };

            // Find up tensor
            let up_key = if tensor_names.contains(&format!("{}.lora_up.weight", base)) {
                format!("{}.lora_up.weight", base)
            } else if tensor_names.contains(&format!("{}.lora_B.weight", base)) {
                format!("{}.lora_B.weight", base)
            } else if tensor_names.contains(&format!("{}.up.weight", base)) {
                format!("{}.up.weight", base)
            } else {
                continue;
            };

            let down = archive.get_tensor(&down_key, &Device::Cpu, DType::F32)?;
            let up = archive.get_tensor(&up_key, &Device::Cpu, DType::F32)?;

            // Determine rank (inner dimension)
            let rank = down.dim(0).unwrap_or(1);
            let alpha = alphas.get(&base).copied();
            let scale = alpha.unwrap_or(rank as f64) / (rank as f64);

            // Determine target and canonical param path
            let (target, target_param) = resolve_target_and_param(&base);

            pairs.push(LoRAPair {
                name: base,
                target,
                target_param,
                down,
                up,
                alpha,
                rank,
                scale,
            });
        }

        if pairs.is_empty() {
            return Err(LuminaError::Config(format!(
                "No valid LoRA weight pairs found in '{}'",
                path_str
            )));
        }

        Ok(LoadedLoRA {
            path: path_str,
            multiplier,
            pairs,
        })
    }
}

fn resolve_target_and_param(base_name: &str) -> (LoRATarget, String) {
    if base_name.starts_with("lora_unet_") {
        let unet_raw = base_name.strip_prefix("lora_unet_").unwrap();
        let param = translate_kohya_unet_name(unet_raw);
        (LoRATarget::UNet, format!("unet.{}", param))
    } else if base_name.starts_with("lora_te1_") {
        let te_raw = base_name.strip_prefix("lora_te1_").unwrap();
        let param = translate_kohya_te_name(te_raw);
        (LoRATarget::ClipL, format!("te1.{}", param))
    } else if base_name.starts_with("lora_te2_") {
        let te_raw = base_name.strip_prefix("lora_te2_").unwrap();
        let param = translate_kohya_te_name(te_raw);
        (LoRATarget::ClipG, format!("te2.{}", param))
    } else if base_name.starts_with("unet.") {
        let param = format!("unet.{}.weight", base_name.strip_prefix("unet.").unwrap());
        (LoRATarget::UNet, param)
    } else if base_name.starts_with("text_encoder.") {
        let param = format!("te1.{}.weight", base_name.strip_prefix("text_encoder.").unwrap());
        (LoRATarget::ClipL, param)
    } else if base_name.starts_with("text_encoder_2.") {
        let param = format!("te2.{}.weight", base_name.strip_prefix("text_encoder_2.").unwrap());
        (LoRATarget::ClipG, param)
    } else {
        (LoRATarget::UNet, format!("unet.{}.weight", base_name))
    }
}

fn translate_kohya_unet_name(name: &str) -> String {
    let mut s = name.to_string();

    // Input blocks
    s = s.replace("input_blocks_0_0", "conv_in");
    s = s.replace("input_blocks_1_0", "down_blocks.0.resnets.0");
    s = s.replace("input_blocks_2_0", "down_blocks.0.resnets.1");
    s = s.replace("input_blocks_3_0_op", "down_blocks.0.downsamplers.0.conv");
    s = s.replace("input_blocks_3_0", "down_blocks.0.downsamplers.0.conv");

    s = s.replace("input_blocks_4_0", "down_blocks.1.resnets.0");
    s = s.replace("input_blocks_4_1", "down_blocks.1.attentions.0");
    s = s.replace("input_blocks_5_0", "down_blocks.1.resnets.1");
    s = s.replace("input_blocks_5_1", "down_blocks.1.attentions.1");
    s = s.replace("input_blocks_6_0_op", "down_blocks.1.downsamplers.0.conv");
    s = s.replace("input_blocks_6_0", "down_blocks.1.downsamplers.0.conv");

    s = s.replace("input_blocks_7_0", "down_blocks.2.resnets.0");
    s = s.replace("input_blocks_7_1", "down_blocks.2.attentions.0");
    s = s.replace("input_blocks_8_0", "down_blocks.2.resnets.1");
    s = s.replace("input_blocks_8_1", "down_blocks.2.attentions.1");

    // Middle block
    s = s.replace("middle_block_0", "mid_block.resnets.0");
    s = s.replace("middle_block_1", "mid_block.attentions.0");
    s = s.replace("middle_block_2", "mid_block.resnets.1");

    // Output blocks
    s = s.replace("output_blocks_0_0", "up_blocks.0.resnets.0");
    s = s.replace("output_blocks_0_1", "up_blocks.0.attentions.0");
    s = s.replace("output_blocks_1_0", "up_blocks.0.resnets.1");
    s = s.replace("output_blocks_1_1", "up_blocks.0.attentions.1");
    s = s.replace("output_blocks_2_0", "up_blocks.0.resnets.2");
    s = s.replace("output_blocks_2_1", "up_blocks.0.attentions.2");
    s = s.replace("output_blocks_2_2_conv", "up_blocks.0.upsamplers.0.conv");
    s = s.replace("output_blocks_2_2", "up_blocks.0.upsamplers.0.conv");

    s = s.replace("output_blocks_3_0", "up_blocks.1.resnets.0");
    s = s.replace("output_blocks_3_1", "up_blocks.1.attentions.0");
    s = s.replace("output_blocks_4_0", "up_blocks.1.resnets.1");
    s = s.replace("output_blocks_4_1", "up_blocks.1.attentions.1");
    s = s.replace("output_blocks_5_0", "up_blocks.1.resnets.2");
    s = s.replace("output_blocks_5_1", "up_blocks.1.attentions.2");
    s = s.replace("output_blocks_5_2_conv", "up_blocks.1.upsamplers.0.conv");
    s = s.replace("output_blocks_5_2", "up_blocks.1.upsamplers.0.conv");

    s = s.replace("output_blocks_6_0", "up_blocks.2.resnets.0");
    s = s.replace("output_blocks_7_0", "up_blocks.2.resnets.1");
    s = s.replace("output_blocks_8_0", "up_blocks.2.resnets.2");

    // Attention sub-blocks
    s = s.replace("_transformer_blocks_", ".transformer_blocks.");
    s = s.replace("_attn1_to_q", ".attn1.to_q");
    s = s.replace("_attn1_to_k", ".attn1.to_k");
    s = s.replace("_attn1_to_v", ".attn1.to_v");
    s = s.replace("_attn1_to_out_0", ".attn1.to_out.0");
    s = s.replace("_attn2_to_q", ".attn2.to_q");
    s = s.replace("_attn2_to_k", ".attn2.to_k");
    s = s.replace("_attn2_to_v", ".attn2.to_v");
    s = s.replace("_attn2_to_out_0", ".attn2.to_out.0");
    s = s.replace("_ff_net_0_proj", ".ff.net_0_proj");
    s = s.replace("_ff_net_2", ".ff.net_2");
    s = s.replace("_proj_in", ".proj_in");
    s = s.replace("_proj_out", ".proj_out");

    // Resnet internal layers
    s = s.replace("_in_layers_2", ".conv1");
    s = s.replace("_out_layers_3", ".conv2");
    s = s.replace("_emb_layers_1", ".time_emb_proj");
    s = s.replace("_skip_connection", ".conv_shortcut");

    format!("{}.weight", s)
}

fn translate_kohya_te_name(name: &str) -> String {
    let mut s = name.to_string();
    s = s.replace("text_model_encoder_layers_", "text_model.encoder.layers.");
    s = s.replace("_self_attn_q_proj", ".self_attn.q_proj");
    s = s.replace("_self_attn_k_proj", ".self_attn.k_proj");
    s = s.replace("_self_attn_v_proj", ".self_attn.v_proj");
    s = s.replace("_self_attn_out_proj", ".self_attn.out_proj");
    s = s.replace("_mlp_fc1", ".mlp.fc1");
    s = s.replace("_mlp_fc2", ".mlp.fc2");

    format!("{}.weight", s)
}
