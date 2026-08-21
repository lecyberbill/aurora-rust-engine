// [WFGY] Zone: SAFE | λ: 0.25 | Fallbacks: 1 (Conv2d reshape fallback) | Action: High performance LoRA hot weight computation & matrix merging

use candle_core::{Result, Tensor};
use super::types::LoRAPair;

pub struct LoRAMerger;

impl LoRAMerger {
    /// Computes the delta weight matrix: \Delta W = multiplier * (alpha / rank) * (up @ down)
    pub fn compute_delta(pair: &LoRAPair, multiplier: f64) -> Result<Tensor> {
        let eff_scale = multiplier * pair.scale;

        let up = &pair.up;
        let down = &pair.down;

        let up_dims = up.shape().dims();
        let down_dims = down.shape().dims();

        if up_dims.len() == 2 && down_dims.len() == 2 {
            // Standard 2D Linear Layer: [d_out, r] @ [r, d_in] -> [d_out, d_in]
            let delta = up.matmul(down)?;
            delta * eff_scale
        } else if up_dims.len() == 4 && down_dims.len() == 4 {
            // 4D Conv2d Layer: [d_out, r, 1, 1] @ [r, d_in, k, k]
            let d_out = up_dims[0];
            let r = up_dims[1];
            let d_in = down_dims[1];
            let kh = down_dims[2];
            let kw = down_dims[3];

            let up_2d = up.reshape((d_out, r))?;
            let down_2d = down.reshape((r, d_in * kh * kw))?;
            let delta_2d = up_2d.matmul(&down_2d)?;
            let delta = delta_2d.reshape((d_out, d_in, kh, kw))?;
            delta * eff_scale
        } else {
            // Generic 2D flatten fallback
            let d_out = up_dims[0];
            let d_in: usize = down_dims.iter().skip(1).copied().product();
            let r = up_dims.iter().skip(1).copied().product::<usize>().min(down_dims[0]);

            let up_flat = up.reshape((d_out, r))?;
            let down_flat = down.reshape((r, d_in))?;
            let delta = up_flat.matmul(&down_flat)?;
            delta * eff_scale
        }
    }
}
