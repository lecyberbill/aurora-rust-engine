// [WFGY] Zone: SAFE | λ: 0.25 | Fallbacks: 0 | Action: High-Performance Pure Rust DPM-Solver++ 2M Karras Scheduler

use candle_core::{Result, Tensor};
use super::Scheduler;
use super::euler::PredictionType;

#[derive(Debug, Clone)]
pub struct DPMSolverMultistepConfig {
    pub num_train_timesteps: usize,
    pub beta_start: f64,
    pub beta_end: f64,
    pub prediction_type: PredictionType,
    pub use_karras_sigmas: bool,
    pub solver_order: usize,
}

impl Default for DPMSolverMultistepConfig {
    fn default() -> Self {
        Self {
            num_train_timesteps: 1000,
            beta_start: 0.00085,
            beta_end: 0.012,
            prediction_type: PredictionType::Epsilon,
            use_karras_sigmas: true,
            solver_order: 2, // 2nd-order Multistep DPM++ (fastest convergence in 18-20 steps)
        }
    }
}

/// DPM-Solver++ (2M) Multistep Discrete Scheduler in Pure Rust
/// Provides state-of-the-art fast ODE convergence, reaching Euler 30-step quality in only 18-20 steps.
pub struct DPMSolverMultistepScheduler {
    config: DPMSolverMultistepConfig,
    _betas: Vec<f64>,
    _alphas_cumprod: Vec<f64>,
    sigmas: Vec<f64>,
    timesteps: Vec<usize>,
    step_sigmas: Vec<f64>,
    model_outputs: Vec<Tensor>, // Cache previous model outputs for 2nd order multistep integration
}

impl DPMSolverMultistepScheduler {
    pub fn new(config: DPMSolverMultistepConfig) -> Self {
        let start = config.beta_start.sqrt();
        let end = config.beta_end.sqrt();
        let step = (end - start) / (config.num_train_timesteps - 1) as f64;
        let betas: Vec<f64> = (0..config.num_train_timesteps)
            .map(|i| {
                let val = start + i as f64 * step;
                val * val
            })
            .collect();

        let mut alphas_cumprod = Vec::with_capacity(config.num_train_timesteps);
        let mut cumprod = 1.0;
        for &b in &betas {
            cumprod *= 1.0 - b;
            alphas_cumprod.push(cumprod);
        }

        let sigmas: Vec<f64> = alphas_cumprod
            .iter()
            .map(|&alpha| ((1.0 - alpha) / alpha).sqrt())
            .collect();

        Self {
            config,
            _betas: betas,
            _alphas_cumprod: alphas_cumprod,
            sigmas,
            timesteps: Vec::new(),
            step_sigmas: Vec::new(),
            model_outputs: Vec::new(),
        }
    }

    fn get_karras_sigmas(&self, num_steps: usize) -> Vec<f64> {
        let sigma_min = *self.sigmas.first().unwrap_or(&0.0001);
        let sigma_max = *self.sigmas.last().unwrap_or(&1.0);
        let rho = 7.0;

        let ramp: Vec<f64> = (0..num_steps)
            .map(|i| i as f64 / (num_steps - 1).max(1) as f64)
            .collect();

        let min_inv_rho = sigma_min.powf(1.0 / rho);
        let max_inv_rho = sigma_max.powf(1.0 / rho);

        let mut sigmas = Vec::with_capacity(num_steps + 1);
        for &r in &ramp {
            let s = (max_inv_rho + r * (min_inv_rho - max_inv_rho)).powf(rho);
            sigmas.push(s);
        }
        sigmas.push(0.0);
        sigmas
    }

    fn index_for_timestep(&self, timestep: usize) -> Option<usize> {
        self.timesteps.iter().position(|&t| t == timestep)
    }

    pub fn sigmas(&self) -> &[f64] {
        &self.step_sigmas
    }
}

impl Scheduler for DPMSolverMultistepScheduler {
    fn set_timesteps(&mut self, num_steps: usize) -> Result<()> {
        let (timesteps, step_sigmas) = if self.config.use_karras_sigmas {
            let karras_sigmas = self.get_karras_sigmas(num_steps);
            let mut matched_timesteps = Vec::with_capacity(num_steps);
            for &s in &karras_sigmas[..num_steps] {
                let mut best_t = 0;
                let mut best_diff = f64::MAX;
                for (t, &train_sigma) in self.sigmas.iter().enumerate() {
                    let diff = (train_sigma - s).abs();
                    if diff < best_diff {
                        best_diff = diff;
                        best_t = t;
                    }
                }
                matched_timesteps.push(best_t);
            }
            (matched_timesteps, karras_sigmas)
        } else {
            let step_ratio = self.config.num_train_timesteps as f64 / num_steps as f64;
            let mut timesteps = Vec::with_capacity(num_steps);
            for i in 0..num_steps {
                let t = (self.config.num_train_timesteps as f64 - 1.0 - (i as f64 * step_ratio)).round() as usize;
                timesteps.push(t.min(self.config.num_train_timesteps - 1));
            }
            let mut sigmas: Vec<f64> = timesteps
                .iter()
                .map(|&t| self.sigmas[t])
                .collect();
            sigmas.push(0.0);
            (timesteps, sigmas)
        };

        self.timesteps = timesteps;
        self.step_sigmas = step_sigmas;
        self.model_outputs.clear();
        Ok(())
    }

    fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    fn sigmas(&self) -> &[f64] {
        &self.step_sigmas
    }

    fn scale_model_input(&self, sample: &Tensor, timestep: usize) -> Result<Tensor> {
        let idx = self.index_for_timestep(timestep).unwrap_or(0);
        let sigma = self.step_sigmas[idx];
        let scale = 1.0 / (sigma.powi(2) + 1.0).sqrt();
        sample.affine(scale, 0.0)
    }

    fn step(
        &mut self,
        model_output: &Tensor,
        timestep: usize,
        sample: &Tensor,
    ) -> Result<Tensor> {
        let step_idx = self
            .index_for_timestep(timestep)
            .ok_or_else(|| candle_core::Error::Msg(format!("Timestep {} not found in schedule", timestep)))?;

        let sigma_t = self.step_sigmas[step_idx];
        let sigma_s = self.step_sigmas[step_idx + 1];

        // 1. Convert model output to predicted original sample (x0)
        let pred_original_sample = match self.config.prediction_type {
            PredictionType::Epsilon => {
                let scaled_noise = model_output.affine(sigma_t, 0.0)?;
                sample.sub(&scaled_noise)?
            }
            PredictionType::VZero => {
                let alpha_prod = 1.0 / (sigma_t.powi(2) + 1.0);
                let alpha = alpha_prod.sqrt();
                let beta = (1.0 - alpha_prod).sqrt();
                let term1 = sample.affine(alpha, 0.0)?;
                let term2 = model_output.affine(beta, 0.0)?;
                term1.sub(&term2)?
            }
            PredictionType::Sample => model_output.clone(),
        };

        // Cache model output for multistep solver
        self.model_outputs.push(pred_original_sample.clone());
        if self.model_outputs.len() > self.config.solver_order {
            self.model_outputs.remove(0);
        }

        // 2. Multistep integration (Order 1 for first step, Order 2 thereafter)
        let lambda_t = -sigma_t.ln();
        let lambda_s = if sigma_s > 1e-6 { -sigma_s.ln() } else { -1e-6f64.ln() };
        let h = lambda_s - lambda_t;

        if self.model_outputs.len() < 2 || sigma_s <= 1e-6 {
            // First order Euler / DPM-Solver-1 step
            // x_{t-1} = (sigma_s / sigma_t) * x_t - alpha_s * (exp(-h) - 1) * pred_x0
            let coeff_sample = sigma_s / sigma_t;
            let coeff_pred = (sigma_s / sigma_t) - 1.0;
            let term1 = sample.affine(coeff_sample, 0.0)?;
            let term2 = pred_original_sample.affine(coeff_pred, 0.0)?;
            term1.sub(&term2)
        } else {
            // Second order 2M Multistep DPM-Solver++ step
            let prev_sigma = self.step_sigmas[step_idx - 1];
            let prev_lambda = -prev_sigma.ln();
            let h_0 = lambda_t - prev_lambda;
            let r0 = h_0 / h;

            let d0 = &self.model_outputs[self.model_outputs.len() - 1];
            let d1 = &self.model_outputs[self.model_outputs.len() - 2];

            // DPM-Solver-2M formula: D = (1 + 1/(2*r0)) * d0 - (1/(2*r0)) * d1
            let w0 = 1.0 + 1.0 / (2.0 * r0);
            let w1 = 1.0 / (2.0 * r0);

            let term_d0 = d0.affine(w0, 0.0)?;
            let term_d1 = d1.affine(w1, 0.0)?;
            let d_effective = term_d0.sub(&term_d1)?;

            let coeff_sample = sigma_s / sigma_t;
            let coeff_pred = (sigma_s / sigma_t) - 1.0;
            let term1 = sample.affine(coeff_sample, 0.0)?;
            let term2 = d_effective.affine(coeff_pred, 0.0)?;
            term1.sub(&term2)
        }
    }
}
