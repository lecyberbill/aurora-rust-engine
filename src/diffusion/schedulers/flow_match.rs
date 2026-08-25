// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: Pure Rust Flow Matching Euler Scheduler for Flux.1 and SD 3.5

use candle_core::{Result, Tensor};
use crate::diffusion::schedulers::Scheduler;

/// Configuration for Flow Matching Euler discrete ODE solver.
#[derive(Debug, Clone)]
pub struct FlowMatchEulerConfig {
    pub shift: f64,
    pub base_shift: f64,
    pub max_shift: f64,
    pub min_shift: f64,
}

impl Default for FlowMatchEulerConfig {
    fn default() -> Self {
        Self {
            shift: 3.0, // Standard shift for Flux.1 Schnell / Dev
            base_shift: 0.5,
            max_shift: 1.15,
            min_shift: 0.5,
        }
    }
}

/// Pure Rust Flow Matching Euler discrete scheduler for Rectified Flow models.
#[derive(Debug, Clone)]
pub struct FlowMatchEulerScheduler {
    config: FlowMatchEulerConfig,
    timesteps: Vec<usize>,
    sigmas: Vec<f64>,
    step_index: usize,
}

impl FlowMatchEulerScheduler {
    pub fn new(config: FlowMatchEulerConfig) -> Self {
        Self {
            config,
            timesteps: Vec::new(),
            sigmas: Vec::new(),
            step_index: 0,
        }
    }

    /// Set inference steps and precompute shifted linear sigma schedule matching official Flux sampling:
    /// mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
    /// sigma_t = exp(mu) / (exp(mu) + (1/t - 1))
    pub fn set_timesteps(&mut self, num_steps: usize) -> Result<()> {
        self.set_timesteps_with_seq_len(num_steps, 4096)
    }

    pub fn set_timesteps_with_seq_len(&mut self, num_steps: usize, image_seq_len: usize) -> Result<()> {
        self.step_index = 0;
        let mut sigmas = Vec::with_capacity(num_steps + 1);

        // Linear interpolation for mu between (256, base_shift) and (4096, max_shift)
        let x1: f64 = 256.0;
        let x2: f64 = 4096.0;
        let m = (self.config.max_shift - self.config.base_shift) / (x2 - x1);
        let b = self.config.base_shift - m * x1;
        let mu = (m * (image_seq_len as f64) + b).clamp(self.config.min_shift, self.config.max_shift);
        let exp_mu = mu.exp();

        for i in 0..=num_steps {
            let t = 1.0 - (i as f64 / num_steps as f64);
            if t <= 0.0 {
                sigmas.push(0.0);
            } else if t >= 1.0 {
                sigmas.push(1.0);
            } else {
                let shifted_t = exp_mu / (exp_mu + (1.0 / t - 1.0));
                sigmas.push(shifted_t);
            }
        }

        let mut timesteps = Vec::with_capacity(num_steps);
        for i in 0..num_steps {
            timesteps.push((sigmas[i] * 1000.0) as usize);
        }

        self.sigmas = sigmas;
        self.timesteps = timesteps;
        Ok(())
    }

    /// Single ODE integration step:
    /// x_{t-1} = x_t + (sigma_{next} - sigma_curr) * model_velocity
    pub fn step(&mut self, model_output: &Tensor, _timestep: usize, sample: &Tensor) -> Result<Tensor> {
        if self.step_index >= self.sigmas.len() - 1 {
            return Ok(sample.clone());
        }

        let sigma_curr = self.sigmas[self.step_index];
        let sigma_next = self.sigmas[self.step_index + 1];
        let dt = sigma_next - sigma_curr;

        self.step_index += 1;

        // x_{next} = sample + dt * velocity
        let orig_dtype = sample.dtype();
        let dt_tensor = Tensor::from_slice(&[dt as f32], (1,), sample.device())?.to_dtype(orig_dtype)?;
        let step_delta = model_output.broadcast_mul(&dt_tensor)?;
        (sample + step_delta)
    }
}

impl Scheduler for FlowMatchEulerScheduler {
    fn set_timesteps(&mut self, num_steps: usize) -> Result<()> {
        self.set_timesteps(num_steps)
    }

    fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    fn scale_model_input(&self, sample: &Tensor, _timestep: usize) -> Result<Tensor> {
        Ok(sample.clone())
    }

    fn step(&mut self, model_output: &Tensor, timestep: usize, sample: &Tensor) -> Result<Tensor> {
        self.step(model_output, timestep, sample)
    }

    fn init_noise_sigma(&self) -> f64 {
        1.0
    }

    fn sigmas(&self) -> &[f64] {
        &self.sigmas
    }
}
