// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: LoRA data structures and type definitions

use candle_core::Tensor;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoRATarget {
    UNet,
    ClipL,
    ClipG,
    /// Flux MMDiT transformer (double/single stream blocks). Deltas are keyed with BFL-style names
    /// such as `double_blocks.0.img_attn.qkv.weight` / `single_blocks.0.linear1.weight`.
    Flux,
}

#[derive(Debug, Clone)]
pub struct LoRAPair {
    pub name: String,
    pub target: LoRATarget,
    pub target_param: String,
    pub down: Tensor,
    pub up: Tensor,
    pub alpha: Option<f64>,
    pub rank: usize,
    pub scale: f64,
}

#[derive(Debug, Clone)]
pub struct LoadedLoRA {
    pub path: String,
    pub multiplier: f64,
    pub pairs: Vec<LoRAPair>,
}
