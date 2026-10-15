// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: CogVideoX Official DDIM v-prediction Scheduler with SNR Shift & Zero Terminal SNR
// Invariant: Exact match with diffusers CogVideoXDDIMScheduler, trailing timesteps, v-prediction inversion.

use candle_core::{Result, Tensor};

#[derive(Debug, Clone)]
pub struct CogVideoXDDIMConfig {
    pub num_train_timesteps: usize,
    pub beta_start: f64,
    pub beta_end: f64,
    pub snr_shift_scale: f64,
}

impl Default for CogVideoXDDIMConfig {
    fn default() -> Self {
        Self {
            num_train_timesteps: 1000,
            beta_start: 0.00085,
            beta_end: 0.0120,
            snr_shift_scale: 3.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CogVideoXDDIMScheduler {
    pub config: CogVideoXDDIMConfig,
    pub alphas_cumprod: Vec<f64>,
    pub timesteps: Vec<usize>,
    pub num_inference_steps: usize,
}

impl CogVideoXDDIMScheduler {
    pub fn new(config: CogVideoXDDIMConfig) -> Self {
        let n = config.num_train_timesteps;
        let start_sqrt = config.beta_start.sqrt();
        let end_sqrt = config.beta_end.sqrt();

        // 1. scaled_linear betas
        let mut betas = Vec::with_capacity(n);
        for i in 0..n {
            let frac = i as f64 / (n - 1) as f64;
            let val = start_sqrt + frac * (end_sqrt - start_sqrt);
            betas.push(val * val);
        }

        // 2. alphas_cumprod
        let mut alphas_cumprod = Vec::with_capacity(n);
        let mut cumprod = 1.0f64;
        for &b in &betas {
            cumprod *= 1.0 - b;
            alphas_cumprod.push(cumprod);
        }

        // 3. SNR shift following SD3 / CogVideoX (snr_shift_scale = 3.0)
        let s = config.snr_shift_scale;
        for a in alphas_cumprod.iter_mut() {
            *a = *a / (s + (1.0 - s) * *a);
        }

        // 4. Rescale zero terminal SNR
        let mut alphas_bar_sqrt: Vec<f64> = alphas_cumprod.iter().map(|&a| a.sqrt()).collect();
        let sqrt_0 = alphas_bar_sqrt[0];
        let sqrt_t = *alphas_bar_sqrt.last().unwrap();
        let scale = sqrt_0 / (sqrt_0 - sqrt_t);
        for val in alphas_bar_sqrt.iter_mut() {
            *val = (*val - sqrt_t) * scale;
        }

        for (i, val) in alphas_bar_sqrt.iter().enumerate() {
            alphas_cumprod[i] = (val * val).clamp(0.0, 1.0);
        }

        Self {
            config,
            alphas_cumprod,
            timesteps: Vec::new(),
            num_inference_steps: 0,
        }
    }

    /// Set discrete inference timesteps using "trailing" spacing matching diffusers
    pub fn set_timesteps(&mut self, num_steps: usize) {
        self.num_inference_steps = num_steps;
        let step_ratio = self.config.num_train_timesteps as f64 / num_steps as f64;
        let mut timesteps = Vec::with_capacity(num_steps);

        for i in 0..num_steps {
            let val = (self.config.num_train_timesteps as f64 - (i as f64 * step_ratio)).round() as i64 - 1;
            timesteps.push(val.max(0) as usize);
        }

        self.timesteps = timesteps;
    }

    pub fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    /// Execute single DDIM reverse step with v-prediction
    pub fn step(
        &self,
        model_output: &Tensor,
        step_idx: usize,
        sample: &Tensor,
    ) -> Result<Tensor> {
        let timestep = self.timesteps[step_idx];
        let step_ratio = self.config.num_train_timesteps / self.num_inference_steps.max(1);
        let prev_timestep = timestep as i64 - step_ratio as i64;

        let alpha_prod_t = self.alphas_cumprod[timestep];
        let alpha_prod_t_prev = if prev_timestep >= 0 {
            self.alphas_cumprod[prev_timestep as usize]
        } else {
            1.0 // set_alpha_to_one = true
        };

        let beta_prod_t = (1.0 - alpha_prod_t).max(0.0);

        // 1. Invert v-prediction to predict original sample x_0:
        // pred_original_sample = sqrt(alpha_prod_t) * sample - sqrt(beta_prod_t) * model_output
        let sqrt_alpha_t = alpha_prod_t.sqrt();
        let sqrt_beta_t = beta_prod_t.sqrt();

        let pred_x0 = {
            let term1 = (sample * sqrt_alpha_t)?;
            let term2 = (model_output * sqrt_beta_t)?;
            (&term1 - &term2)?
        };

        // 2. Compute coefficients a_t and b_t
        let a_t = ((1.0 - alpha_prod_t_prev).max(0.0) / (1.0 - alpha_prod_t).max(1e-12)).sqrt();
        let b_t = alpha_prod_t_prev.sqrt() - sqrt_alpha_t * a_t;

        // 3. Compute prev_sample x_{t-1} = a_t * sample + b_t * pred_original_sample
        let prev_sample = ((sample * a_t)? + (&pred_x0 * b_t)?)?;

        Ok(prev_sample)
    }
}
