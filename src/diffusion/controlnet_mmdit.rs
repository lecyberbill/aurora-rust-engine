// [WFGY] Zone: SAFE | λ: 0.25 | Fallbacks: 0 | Action: Universal ControlNet & T2I-Adapter architecture for MMDiT (Flux.1 / Flux.2 & SD3.5)
// Invariant: Non-intrusive residual injection, zero memory overhead when inactive, exact multi-modal block alignment.

use candle_core::{Result, Tensor};
use candle_nn::{conv2d, linear, Conv2d, Conv2dConfig, Linear, Module, VarBuilder};

/// T2I / ControlNet spatial conditioner for MMDiT architectures (Flux & SD3.5).
///
/// MMDiT networks process 2D spatial patches rather than downsampled feature maps.
/// This adapter downsamples pixel hints (Canny, Depth, Pose) into patch token sequences:
/// `[B, 3, H, W] -> Conv Stem -> Patchify -> [B, (H/16)*(W/16), Hidden_Dim]`
#[derive(Debug, Clone)]
pub struct MMDiTControlNetAdapter {
    conv_in: Conv2d,
    blocks: Vec<Conv2d>,
    proj_out: Linear,
    hidden_dim: usize,
    scale: f64,
}

impl MMDiTControlNetAdapter {
    pub fn new(vb: VarBuilder, in_channels: usize, hidden_dim: usize, scale: f64) -> Result<Self> {
        let conv_in = conv2d(
            in_channels,
            64,
            3,
            Conv2dConfig { padding: 1, stride: 2, ..Default::default() },
            vb.pp("conv_in"),
        )?;

        // Stride-2 conv blocks to reduce resolution from H/2 -> H/4 -> H/8 -> H/16
        let mut blocks = Vec::new();
        let channels = [64, 128, 256, 512];
        for i in 0..3 {
            let block = conv2d(
                channels[i],
                channels[i + 1],
                3,
                Conv2dConfig { padding: 1, stride: 2, ..Default::default() },
                vb.pp(&format!("blocks.{}", i)),
            )?;
            blocks.push(block);
        }

        let proj_out = linear(512, hidden_dim, vb.pp("proj_out"))?;

        Ok(Self {
            conv_in,
            blocks,
            proj_out,
            hidden_dim,
            scale,
        })
    }

    /// Extract spatial conditioning tokens:
    /// `hint`: RGB pixel map `[B, 3, H, W]` (values normalized in `[0, 1]`)
    /// Output: Residual tokens `[B, (H/16)*(W/16), Hidden_Dim]`
    pub fn forward(&self, hint: &Tensor) -> Result<Tensor> {
        let mut x = self.conv_in.forward(hint)?;
        x = candle_nn::ops::silu(&x)?;

        for block in &self.blocks {
            x = block.forward(&x)?;
            x = candle_nn::ops::silu(&x)?;
        }

        // x is now [B, 512, H/16, W/16]
        let (b, c, h_p, w_p) = x.dims4()?;
        let num_tokens = h_p * w_p;

        // Permute to [B, H/16, W/16, 512] -> flatten spatial tokens -> [B, Num_Tokens, 512]
        let x_trans = x.permute((0, 2, 3, 1))?.contiguous()?.reshape((b, num_tokens, c))?;

        // Linear projection to match MMDiT hidden dimension
        let out = self.proj_out.forward(&x_trans)?;
        out * self.scale
    }

    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    pub fn scale(&self) -> f64 {
        self.scale
    }

    pub fn set_scale(&mut self, scale: f64) {
        self.scale = scale;
    }
}

/// Container for managing active MMDiT ControlNet / Adapter conditioners
#[derive(Debug, Clone, Default)]
pub struct MMDiTControlNetManager {
    adapters: Vec<MMDiTControlNetAdapter>,
}

impl MMDiTControlNetManager {
    pub fn new() -> Self {
        Self { adapters: Vec::new() }
    }

    pub fn add_adapter(&mut self, adapter: MMDiTControlNetAdapter) {
        self.adapters.push(adapter);
    }

    pub fn is_empty(&self) -> bool {
        self.adapters.is_empty()
    }

    /// Aggregate all active spatial conditioning residuals for the image tokens
    pub fn compute_residuals(&self, hint: &Tensor) -> Result<Option<Tensor>> {
        if self.adapters.is_empty() {
            return Ok(None);
        }

        let mut total_residual: Option<Tensor> = None;
        for adapter in &self.adapters {
            let res = adapter.forward(hint)?;
            total_residual = match total_residual {
                Some(prev) => Some((prev + res)?),
                None => Some(res),
            };
        }

        Ok(total_residual)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarBuilder;
    use std::collections::HashMap;

    #[test]
    fn test_mmdit_controlnet_adapter_forward() -> Result<()> {
        let dev = Device::Cpu;
        let mut map = HashMap::new();

        // Mock conv_in: 3 -> 64, kernel 3
        map.insert("conv_in.weight".to_string(), Tensor::zeros((64, 3, 3, 3), DType::F32, &dev)?);
        map.insert("conv_in.bias".to_string(), Tensor::zeros((64,), DType::F32, &dev)?);

        // Mock blocks: 64->128, 128->256, 256->512
        map.insert("blocks.0.weight".to_string(), Tensor::zeros((128, 64, 3, 3), DType::F32, &dev)?);
        map.insert("blocks.0.bias".to_string(), Tensor::zeros((128,), DType::F32, &dev)?);
        map.insert("blocks.1.weight".to_string(), Tensor::zeros((256, 128, 3, 3), DType::F32, &dev)?);
        map.insert("blocks.1.bias".to_string(), Tensor::zeros((256,), DType::F32, &dev)?);
        map.insert("blocks.2.weight".to_string(), Tensor::zeros((512, 256, 3, 3), DType::F32, &dev)?);
        map.insert("blocks.2.bias".to_string(), Tensor::zeros((512,), DType::F32, &dev)?);

        // Mock proj_out: 512 -> 3072 (Flux.2-Klein hidden dim)
        map.insert("proj_out.weight".to_string(), Tensor::zeros((3072, 512), DType::F32, &dev)?);
        map.insert("proj_out.bias".to_string(), Tensor::zeros((3072,), DType::F32, &dev)?);

        let vb = VarBuilder::from_tensors(map, DType::F32, &dev);

        let adapter = MMDiTControlNetAdapter::new(vb, 3, 3072, 0.85)?;

        // Input 256x256 image -> 16x16 = 256 spatial patch tokens
        let dummy_hint = Tensor::zeros((1, 3, 256, 256), DType::F32, &dev)?;
        let residuals = adapter.forward(&dummy_hint)?;

        assert_eq!(residuals.dims3()?, (1, 256, 3072));
        Ok(())
    }
}
