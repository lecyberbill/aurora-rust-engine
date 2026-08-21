// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: Discrete Euler Scheduler with Karras sigma-timestep alignment

use candle_core::{Result, Tensor};
use super::Scheduler;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PredictionType {
    Epsilon,
    VZero,
    Sample,
}

#[derive(Debug, Clone)]
pub struct EulerSchedulerConfig {
    pub num_train_timesteps: usize,
    pub beta_start: f64,
    pub beta_end: f64,
    pub beta_schedule: BetaSchedule,
    pub prediction_type: PredictionType,
    pub use_karras_sigmas: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BetaSchedule {
    Linear,
    ScaledLinear,
}

impl Default for EulerSchedulerConfig {
    fn default() -> Self {
        Self {
            num_train_timesteps: 1000,
            beta_start: 0.00085,
            beta_end: 0.012,
            beta_schedule: BetaSchedule::ScaledLinear,
            prediction_type: PredictionType::Epsilon,
            use_karras_sigmas: false,
        }
    }
}

pub struct EulerDiscreteScheduler {
    config: EulerSchedulerConfig,
    _betas: Vec<f64>,
    _alphas_cumprod: Vec<f64>,
    sigmas: Vec<f64>,
    timesteps: Vec<usize>,
    step_sigmas: Vec<f64>,
}

impl EulerDiscreteScheduler {
    pub fn new(config: EulerSchedulerConfig) -> Self {
        let betas = match config.beta_schedule {
            BetaSchedule::Linear => {
                let step = (config.beta_end - config.beta_start) / (config.num_train_timesteps - 1) as f64;
                (0..config.num_train_timesteps)
                    .map(|i| config.beta_start + i as f64 * step)
                    .collect::<Vec<_>>()
            }
            BetaSchedule::ScaledLinear => {
                let start = config.beta_start.sqrt();
                let end = config.beta_end.sqrt();
                let step = (end - start) / (config.num_train_timesteps - 1) as f64;
                (0..config.num_train_timesteps)
                    .map(|i| {
                        let val = start + i as f64 * step;
                        val * val
                    })
                    .collect::<Vec<_>>()
            }
        };

        let mut alphas_cumprod = Vec::with_capacity(config.num_train_timesteps);
        let mut cumprod = 1.0;
        for &b in &betas {
            cumprod *= 1.0 - b;
            alphas_cumprod.push(cumprod);
        }

        // Base sigmas from training schedule: sqrt((1 - alpha_prod) / alpha_prod)
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

impl Scheduler for EulerDiscreteScheduler {
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
        Ok(())
    }

    fn timesteps(&self) -> &[usize] {
        &self.timesteps
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

        let sigma = self.step_sigmas[step_idx];
        let sigma_next = self.step_sigmas[step_idx + 1];

        // 1. Compute predicted original sample (x_0)
        let pred_original_sample = match self.config.prediction_type {
            PredictionType::Epsilon => {
                let scaled_noise = model_output.affine(sigma, 0.0)?;
                sample.sub(&scaled_noise)?
            }
            PredictionType::VZero => {
                let alpha_prod = 1.0 / (sigma.powi(2) + 1.0);
                let alpha = alpha_prod.sqrt();
                let beta = (1.0 - alpha_prod).sqrt();
                let term1 = sample.affine(alpha, 0.0)?;
                let term2 = model_output.affine(beta, 0.0)?;
                term1.sub(&term2)?
            }
            PredictionType::Sample => model_output.clone(),
        };

        // 2. Compute 1st order derivative d = (sample - pred_x0) / sigma
        let derivative = if sigma > 1e-6 {
            sample.sub(&pred_original_sample)?.affine(1.0 / sigma, 0.0)?
        } else {
            sample.zeros_like()?
        };

        // 3. Euler step: x_{t-1} = sample + d * (sigma_next - sigma)
        let dt = sigma_next - sigma;
        let delta = derivative.affine(dt, 0.0)?;
        sample.add(&delta)
    }
}
