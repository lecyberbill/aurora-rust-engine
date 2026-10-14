// [WFGY] Zone: SAFE | λ: 0.15 | Fallbacks: 0 | Action: Pipeline Execution Mode and Lifecycle Management
// Invariant: Deterministic resource management supporting persistent VRAM, staged offload, and block streaming.

use serde::{Deserialize, Serialize};

/// Execution mode controlling VRAM retention and sequential model lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PipelineExecutionMode {
    /// All models (text encoder, transformer/DiT, VAEs) remain concurrently resident in VRAM.
    /// Best performance for high-end workstations or server GPUs with sufficient VRAM (e.g. >= 48 GB / 80 GB).
    PersistentVram,

    /// Stage-by-stage sequential offload: loads and unloads model components between pipeline phases:
    /// - Phase 1: Text Conditioning -> VRAM, then offloaded to Host RAM.
    /// - Phase 2: Denoising DiT -> VRAM, runs 4-8 turbo steps, then offloaded.
    /// - Phase 3: Dual VAEs (Video ViT3D + Audio BigVGAN) -> VRAM, decoded and finalized.
    /// Recommended default for 24 GB - 32 GB GPUs (e.g. AMD Radeon AI PRO R9700 32 GB, RTX 3090/4090 24 GB).
    #[default]
    SequentialStageOffload,

    /// Sequential stage offload combined with layer-by-layer block streaming (`SequentialBlockStreamer`).
    /// Enables running massive foundation models (like MiniMax H3 33B or Flux.1) on constrained VRAM (< 16 GB, e.g. RTX 4070 Ti / 3060).
    SequentialBlockStreaming,
}

impl PipelineExecutionMode {
    /// Returns true if stages should be offloaded sequentially rather than held persistently in VRAM.
    pub fn is_sequential(&self) -> bool {
        matches!(self, Self::SequentialStageOffload | Self::SequentialBlockStreaming)
    }

    /// Returns true if layer-by-layer sequential block streaming is enabled for the transformer backbone.
    pub fn is_block_streaming(&self) -> bool {
        matches!(self, Self::SequentialBlockStreaming)
    }

    /// Target peak VRAM budget in megabytes for this execution mode.
    pub fn target_peak_vram_mb(&self) -> usize {
        match self {
            Self::PersistentVram => 49152,          // 48 GB+
            Self::SequentialStageOffload => 32768,  // 32 GB
            Self::SequentialBlockStreaming => 12288,// 12 GB
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_execution_mode_invariants() {
        let default_mode = PipelineExecutionMode::default();
        assert_eq!(default_mode, PipelineExecutionMode::SequentialStageOffload);
        assert!(default_mode.is_sequential());
        assert!(!default_mode.is_block_streaming());
        assert_eq!(default_mode.target_peak_vram_mb(), 32768);

        let persistent = PipelineExecutionMode::PersistentVram;
        assert!(!persistent.is_sequential());
        assert!(!persistent.is_block_streaming());
        assert_eq!(persistent.target_peak_vram_mb(), 49152);

        let streaming = PipelineExecutionMode::SequentialBlockStreaming;
        assert!(streaming.is_sequential());
        assert!(streaming.is_block_streaming());
        assert_eq!(streaming.target_peak_vram_mb(), 12288);
    }
}
