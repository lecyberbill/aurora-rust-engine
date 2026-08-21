// [WFGY] Zone: SAFE | λ: 0.25 | Fallbacks: 0 | Action: DDIM deterministic step scheduler

use candle_core::{Result, Tensor};
use super::Scheduler;

#[derive(Debug, Clone)]
pub struct DDIMConfig {
    pub num_train_timesteps: usize,
    pub beta_start: f64,
    pub beta_end: f64,
    pub eta: f64,
}

impl Default for DDIMConfig {
    fn default() -> Self {
        Self {
            num_train_timesteps: 1000,
            beta_start: 0.00085,
            beta_end: 0.012,
            eta: 0.0,
        }
    }
}

pub struct DDIMScheduler {
    config: DDIMConfig,
    alphas_cumprod: Vec<f64>,
    timesteps: Vec<usize>,
}

impl DDIMScheduler {
    pub fn new(config: DDIMConfig) -> Self {
        let start = config.beta_start.sqrt();
        let end = config.beta_end.sqrt();
        let step = (end - start) / (config.num_train_timesteps - 1) as f64;
        let betas: Vec<f64> = (0..config.num_train_timesteps)
            .map(|i| {
                let v = start + i as f64 * step;
                v * v
            })
            .collect();

        let mut alphas_cumprod = Vec::with_capacity(config.num_train_timesteps);
        let mut cumprod = 1.0;
        for &b in &betas {
            cumprod *= 1.0 - b;
            alphas_cumprod.push(cumprod);
        }

        Self {
            config,
            alphas_cumprod,
            timesteps: Vec::new(),
        }
    }

    fn index_for_timestep(&self, timestep: usize) -> Option<usize> {
        self.timesteps.iter().position(|&t| t == timestep)
    }
}

impl Scheduler for DDIMScheduler {
    fn set_timesteps(&mut self, num_steps: usize) -> Result<()> {
        let step_ratio = self.config.num_train_timesteps as f64 / num_steps as f64;
        let mut timesteps = Vec::with_capacity(num_steps);
        for i in 0..num_steps {
            let t = (self.config.num_train_timesteps as f64 - 1.0 - (i as f64 * step_ratio)).round() as usize;
            timesteps.push(t.min(self.config.num_train_timesteps - 1));
        }
        self.timesteps = timesteps;
        Ok(())
    }

    fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    fn scale_model_input(&self, sample: &Tensor, _timestep: usize) -> Result<Tensor> {
        // DDIM uses raw sample
        Ok(sample.clone())
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

        let alpha_prod_t = self.alphas_cumprod[timestep];
        let prev_timestep = if step_idx + 1 < self.timesteps.len() {
            self.timesteps[step_idx + 1]
        } else {
            0
        };
        let alpha_prod_t_prev = if prev_timestep > 0 {
            self.alphas_cumprod[prev_timestep]
        } else {
            1.0
        };

        let beta_prod_t = 1.0 - alpha_prod_t;
        let beta_prod_t_prev = 1.0 - alpha_prod_t_prev;

        // 1. Predict original sample x_0
        let pred_original_sample = {
            let term1 = sample.affine(1.0 / alpha_prod_t.sqrt(), 0.0)?;
            let term2 = model_output.affine(beta_prod_t.sqrt() / alpha_prod_t.sqrt(), 0.0)?;
            term1.sub(&term2)?
        };

        // 2. Compute variance sigma_t for non-deterministic DDIM (eta > 0)
        let variance = (beta_prod_t_prev / beta_prod_t) * (1.0 - alpha_prod_t / alpha_prod_t_prev);
        let std_dev_t = self.config.eta * variance.max(0.0).sqrt();

        // 3. Compute direction pointing to x_t
        let pred_sample_direction_coeff = (1.0 - alpha_prod_t_prev - std_dev_t.powi(2)).max(0.0).sqrt();
        let pred_sample_direction = model_output.affine(pred_sample_direction_coeff, 0.0)?;

        // 4. Compute x_{t-1}
        let prev_sample = pred_original_sample
            .affine(alpha_prod_t_prev.sqrt(), 0.0)?
            .add(&pred_sample_direction)?;

        Ok(prev_sample)
    }
}
