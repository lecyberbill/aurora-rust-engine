// [WFGY] Zone: SAFE | λ: 0.20 | Action: Dedicated MoE (Mixture of Experts) Router & Dispatch
//
// Modular implementation supporting DeepSeek V3 / R1 and Mixtral.

use candle_core::Tensor;
use crate::error::{LuminaError, Result};

/// MoE Dispatch Configuration & Router Logic
pub struct MoeRouter {
    pub num_experts: usize,
    pub top_k: usize,
}

impl MoeRouter {
    pub fn new(num_experts: usize, top_k: usize) -> Self {
        Self {
            num_experts,
            top_k,
        }
    }

    /// Computes top-k expert assignments and normalized routing weights for each token.
    ///
    /// Returns: Vec<Vec<(expert_idx, normalized_weight)>> for each token in batch
    pub fn route(
        &self,
        router_probs: &Tensor,
        total_tokens: usize,
    ) -> Result<Vec<Vec<(usize, f32)>>> {
        let prob_flat: Vec<f32> = router_probs
            .flatten_all()
            .map_err(LuminaError::Candle)?
            .to_vec1()
            .map_err(LuminaError::Candle)?;

        let mut token_routings = Vec::with_capacity(total_tokens);

        for tok_idx in 0..total_tokens {
            let start = tok_idx * self.num_experts;
            let end = (start + self.num_experts).min(prob_flat.len());
            let slice = &prob_flat[start..end];

            let mut indexed: Vec<(usize, f32)> = slice
                .iter()
                .copied()
                .enumerate()
                .collect();

            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            indexed.truncate(self.top_k);

            let weight_sum: f32 = indexed.iter().map(|(_, w)| *w).sum();
            let norm_factor = if weight_sum > 0.0 { 1.0 / weight_sum } else { 1.0 };

            let normalized: Vec<(usize, f32)> = indexed
                .into_iter()
                .map(|(idx, w)| (idx, w * norm_factor))
                .collect();

            token_routings.push(normalized);
        }

        Ok(token_routings)
    }
}
