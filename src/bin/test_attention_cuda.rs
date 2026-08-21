// [WFGY] Zone: SAFE | λ: 0.15 | Fallbacks: 0 | Action: UNet Attention forward tester on CUDA with Some(64)

use candle_core::{DType, Device, Tensor};
use candle_nn::VarMap;
use candle_transformers::models::stable_diffusion::unet_2d::{
    BlockConfig, UNet2DConditionModel, UNet2DConditionModelConfig,
};

fn main() -> anyhow::Result<()> {
    let device = Device::new_cuda(0)?;
    let varmap = VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F16, &device);

    for slice_size in [Some(64), Some(640)] {
        println!("Testing slice_size on CUDA: {:?}", slice_size);
        let config = UNet2DConditionModelConfig {
            blocks: vec![
                BlockConfig {
                    out_channels: 320,
                    use_cross_attn: None,
                    attention_head_dim: 64,
                },
                BlockConfig {
                    out_channels: 640,
                    use_cross_attn: Some(2),
                    attention_head_dim: 64,
                },
                BlockConfig {
                    out_channels: 1280,
                    use_cross_attn: Some(10),
                    attention_head_dim: 64,
                },
            ],
            center_input_sample: false,
            cross_attention_dim: 2048,
            downsample_padding: 1,
            flip_sin_to_cos: true,
            freq_shift: 0.0,
            layers_per_block: 2,
            mid_block_scale_factor: 1.0,
            norm_eps: 1e-5,
            norm_num_groups: 32,
            sliced_attention_size: slice_size,
            use_linear_projection: true,
        };

        match UNet2DConditionModel::new(vb.clone(), 4, 4, true, config) {
            Ok(unet) => {
                let sample = Tensor::randn(0f32, 1f32, (2, 4, 64, 64), &device)?.to_dtype(DType::F16)?;
                let text = Tensor::randn(0f32, 1f32, (2, 77, 2048), &device)?.to_dtype(DType::F16)?;
                match unet.forward(&sample, 1.0, &text) {
                    Ok(out) => println!("  ✅ CUDA Forward Success! Shape: {:?}", out.shape()),
                    Err(e) => println!("  ❌ CUDA Forward Error: {:?}", e),
                }
            }
            Err(e) => println!("  ❌ Init Error: {:?}", e),
        }
    }
    Ok(())
}
