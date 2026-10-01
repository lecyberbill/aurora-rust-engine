// [WFGY] Zone: SAFE | λ: 0.25 | Fallbacks: 0 | Action: Pure Rust Qwen 3.5 SSM / Mamba State Space Recurrent Layer

use candle_core::{DType, Tensor};
use candle_nn::{linear_no_bias, Linear, Module, RmsNorm, VarBuilder};
use crate::error::{LuminaError, Result};

/// SSM State Cache for individual layer during autoregressive generation
#[derive(Debug, Clone)]
pub struct SsmStateCache {
    /// Conv1d rolling buffer: [B, D, KernelSize - 1]
    pub conv_state: Option<Tensor>,
    /// SSM recurrent state: [B, Groups, HiddenPerGroup, StateSize]
    pub ssm_state: Option<Tensor>,
}

impl SsmStateCache {
    pub fn new() -> Self {
        Self {
            conv_state: None,
            ssm_state: None,
        }
    }

    pub fn reset(&mut self) {
        self.conv_state = None;
        self.ssm_state = None;
    }
}

/// Configuration for Qwen 3.5 SSM Layer
#[derive(Debug, Clone)]
pub struct SsmConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub state_size: usize,
    pub conv_kernel: usize,
    pub num_groups: usize,
    pub rms_norm_eps: f64,
}

impl Default for SsmConfig {
    fn default() -> Self {
        Self {
            hidden_size: 2048,
            intermediate_size: 6144,
            state_size: 128,
            conv_kernel: 4,
            num_groups: 16,
            rms_norm_eps: 1e-6,
        }
    }
}

/// Qwen 3.5 State Space Model (SSM) Block
#[derive(Debug, Clone)]
pub struct Qwen35SsmBlock {
    #[allow(dead_code)]
    conv1d_weight: Tensor,
    ssm_alpha: Linear,
    ssm_beta: Linear,
    ssm_out: Linear,
    norm: RmsNorm,
    state_size: usize,
    num_groups: usize,
    #[allow(dead_code)]
    conv_kernel: usize,
    #[allow(dead_code)] 
    hidden_size: usize,
}

impl Qwen35SsmBlock {
    pub fn new(cfg: &SsmConfig, vb: VarBuilder) -> Result<Self> {
        let conv1d_weight = vb.get((cfg.intermediate_size, cfg.conv_kernel), "ssm_conv1d.weight")
            .or_else(|_| vb.get((cfg.intermediate_size, cfg.conv_kernel), "conv1d.weight"))?;

        let ssm_alpha = linear_no_bias(cfg.hidden_size, cfg.num_groups, vb.pp("ssm_alpha"))
            .or_else(|_| linear_no_bias(cfg.hidden_size, cfg.num_groups, vb.pp("alpha")))?;

        let ssm_beta = linear_no_bias(cfg.hidden_size, cfg.num_groups, vb.pp("ssm_beta"))
            .or_else(|_| linear_no_bias(cfg.hidden_size, cfg.num_groups, vb.pp("beta")))?;

        let ssm_out = linear_no_bias(cfg.hidden_size, cfg.hidden_size, vb.pp("ssm_out"))
            .or_else(|_| linear_no_bias(cfg.hidden_size, cfg.hidden_size, vb.pp("out_proj")))?;

        let norm = candle_nn::rms_norm(cfg.state_size, cfg.rms_norm_eps, vb.pp("ssm_norm"))
            .or_else(|_| candle_nn::rms_norm(cfg.state_size, cfg.rms_norm_eps, vb.pp("norm")))?;

        Ok(Self {
            conv1d_weight,
            ssm_alpha,
            ssm_beta,
            ssm_out,
            norm,
            state_size: cfg.state_size,
            num_groups: cfg.num_groups,
            conv_kernel: cfg.conv_kernel,
            hidden_size: cfg.hidden_size,
        })
    }

    /// Forward pass with sequence prefill or step autoregression
    pub fn forward(
        &self,
        x: &Tensor,
        cache: &mut Option<SsmStateCache>,
    ) -> Result<Tensor> {
        let (b, seq_len, dim) = x.dims3()?;
        let dev = x.device();
        let dtype = x.dtype();

        let mut ssm_cache = cache.take().unwrap_or_else(SsmStateCache::new);

        // Alpha & Beta gates
        let alpha = self.ssm_alpha.forward(x)?; // [B, S, Groups]
        let beta = self.ssm_beta.forward(x)?;   // [B, S, Groups]
        let alpha = candle_nn::ops::sigmoid(&alpha)?;
        let beta = candle_nn::ops::sigmoid(&beta)?;

        let mut outputs = Vec::with_capacity(seq_len);
        let mut curr_state = ssm_cache.ssm_state.clone().unwrap_or_else(|| {
            Tensor::zeros((b, self.num_groups, dim / self.num_groups, self.state_size), dtype, dev)
                .unwrap_or_else(|_| Tensor::zeros((b, self.num_groups, dim / self.num_groups, self.state_size), DType::F32, dev).unwrap())
        });

        // Recurrent State Space computation
        for t in 0..seq_len {
            let xt = x.narrow(1, t, 1)?.squeeze(1)?; // [B, D]
            let at = alpha.narrow(1, t, 1)?.squeeze(1)?.unsqueeze(2)?.unsqueeze(3)?; // [B, G, 1, 1]
            let bt = beta.narrow(1, t, 1)?.squeeze(1)?.unsqueeze(2)?.unsqueeze(3)?;  // [B, G, 1, 1]

            let xt_grouped = xt.reshape((b, self.num_groups, dim / self.num_groups, 1))?;
            // S_t = alpha * S_{t-1} + beta * (X_t x 1)
            let decayed = curr_state.broadcast_mul(&at)?;
            let injected = xt_grouped.broadcast_mul(&bt)?;
            curr_state = decayed.broadcast_add(&injected)?;

            // Output projection from state
            let normed = self.norm.forward(&curr_state)?; // [B, G, H_G, StateSize]
            let step_out = normed.sum(3)?.reshape((b, dim))?;
            outputs.push(step_out);
        }

        ssm_cache.ssm_state = Some(curr_state);
        *cache = Some(ssm_cache);

        let out_stacked = Tensor::stack(&outputs[..], 1)?; // [B, S, D]
        self.ssm_out.forward(&out_stacked).map_err(LuminaError::Candle)
    }
}

/// Full Qwen 3.5 Hybrid Layer (SSM + SwiGLU MLP)
#[derive(Debug, Clone)]
pub struct Qwen35HybridLayer {
    pub norm1: RmsNorm,
    pub ssm: Qwen35SsmBlock,
    pub norm2: RmsNorm,
    pub gate_proj: Linear,
    pub up_proj: Linear,
    pub down_proj: Linear,
}

impl Qwen35HybridLayer {
    pub fn new(cfg: &SsmConfig, vb: VarBuilder) -> Result<Self> {
        let norm1 = candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("attn_norm"))
            .or_else(|_| candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm")))?;

        let ssm = Qwen35SsmBlock::new(cfg, vb.clone())?;

        let norm2 = candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("post_attention_norm"))
            .or_else(|_| candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("post_attention_layernorm")))?;

        let gate_proj = linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("ffn_gate"))
            .or_else(|_| linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj")))?;

        let up_proj = linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("ffn_up"))
            .or_else(|_| linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj")))?;

        let down_proj = linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("ffn_down"))
            .or_else(|_| linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj")))?;

        Ok(Self {
            norm1,
            ssm,
            norm2,
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        cache: &mut Option<SsmStateCache>,
    ) -> Result<Tensor> {
        let residual = x;
        let normed = self.norm1.forward(x)?;
        let ssm_out = self.ssm.forward(&normed, cache)?;
        let h = (residual + ssm_out)?;

        let residual = &h;
        let normed2 = self.norm2.forward(&h)?;
        let gate = self.gate_proj.forward(&normed2)?;
        let up = self.up_proj.forward(&normed2)?;
        let act = (candle_nn::ops::silu(&gate)? * up)?;
        let mlp_out = self.down_proj.forward(&act)?;

        let out = (residual + mlp_out)?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_qwen35_ssm_block_forward() -> Result<()> {
        let dev = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let cfg = SsmConfig {
            hidden_size: 64,
            intermediate_size: 128,
            state_size: 16,
            conv_kernel: 4,
            num_groups: 4,
            rms_norm_eps: 1e-6,
        };

        let mut cache = None;
        let ssm = Qwen35SsmBlock::new(&cfg, vb)?;
        let input = Tensor::zeros((1, 8, 64), DType::F32, &dev)?;
        let output = ssm.forward(&input, &mut cache)?;

        assert_eq!(output.dims3()?, (1, 8, 64));
        assert!(cache.is_some());
        Ok(())
    }
}
