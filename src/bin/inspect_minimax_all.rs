use candle_core::{DType, Device};
use candle_nn::VarBuilder;

fn main() -> anyhow::Result<()> {
    println!("ROCm Device Probing:");
    #[allow(unused_variables)]
    for ord in 0..2 {
        #[cfg(feature = "rocm")]
        match Device::new_rocm(ord) {
            Ok(d) => {
                println!("Device::new_rocm({}) = {:?}", ord, d);
                match candle_core::Tensor::zeros((1024, 1024, 256), DType::F32, &d) {
                    Ok(_) => println!("   Successfully allocated 1GB tensor on device {}!", ord),
                    Err(e) => println!("   Failed 1GB allocation on device {}: {:?}", ord, e),
                }
            }
            Err(e) => println!("Device::new_rocm({}) err: {:?}", ord, e),
        }
    }
    let dev = Device::Cpu;
    let dit_path = "/models/comfyui/minimax_h3/diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors";
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[dit_path], DType::F32, &dev)? };

    println!("1. video_patch_proj: {:?}", candle_nn::linear(96, 5376, vb.pp("video_patch_proj")));
    println!("2. audio_patch_proj: {:?}", candle_nn::linear(32, 5376, vb.pp("audio_patch_proj")));
    println!("3. condition_proj: {:?}", candle_nn::linear(5120, 5376, vb.pp("condition_proj")));
    println!("4. adaln_t_table: {:?}", vb.get((1025, 8), "adaln_t_table"));

    let b0_vb = vb.pp("blocks.0");
    println!("5. blocks.0:");
    let w_norm1 = b0_vb.pp("norm1").get(5376, "weight");
    println!("   norm1.weight: {:?}", w_norm1);
    let w_qnorm = b0_vb.pp("attn").pp("q_norm").get(128, "weight");
    println!("   q_norm.weight: {:?}", w_qnorm);
    println!("   attn.out_proj: {:?}", candle_nn::linear_no_bias(7168, 5376, b0_vb.pp("attn").pp("out_proj")));
    println!("   mlp.fc1: {:?}", candle_nn::linear_no_bias(5376, 14336 * 2, b0_vb.pp("mlp").pp("fc1")));
    println!("   mlp.fc2: {:?}", candle_nn::linear_no_bias(14336, 5376, b0_vb.pp("mlp").pp("fc2")));
    println!("   adaln_proj.linear: {:?}", candle_nn::linear(8, 6 * 5376 * 3, b0_vb.pp("adaln_proj").pp("linear")));

    println!("7. Block 0 loading test in BF16:");
    let mut config = aurora_rust_engine::diffusion::dit::minimax_h3::MiniMaxH3Config::default();
    config.num_layers = 1; // Load only 1 block to measure exact RAM/VRAM
    let vb_bf16 = unsafe { VarBuilder::from_mmaped_safetensors(&[dit_path], DType::BF16, &dev)? };
    let blk0 = aurora_rust_engine::diffusion::dit::minimax_h3::H3DiTBlock::load(
        vb_bf16.pp("blocks.0"),
        config.hidden_size,
        config.num_attention_heads,
        config.attention_head_dim,
        config.ffn_hidden_size,
        8,
        1e-5,
    );
    match blk0 {
        Ok(_) => println!("   Successfully loaded H3DiTBlock 0 in BF16!"),
        Err(e) => println!("   Failed to load block 0: {:?}", e),
    }

    Ok(())
}
