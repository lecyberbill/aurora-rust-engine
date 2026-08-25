// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: Diffusion scheduler trait definition and factory

use candle_core::{Result, Tensor};

pub mod euler;
pub mod ddim;
pub mod dpm_solver;
pub mod flow_match;

pub use euler::{EulerDiscreteScheduler, EulerSchedulerConfig, PredictionType};
pub use ddim::{DDIMScheduler, DDIMConfig};
pub use dpm_solver::{DPMSolverMultistepScheduler, DPMSolverMultistepConfig};
pub use flow_match::{FlowMatchEulerScheduler, FlowMatchEulerConfig};

/// Mathematical scheduler contract for stepping iterative noise diffusion models.
pub trait Scheduler: Send + Sync {
    /// Initialize timesteps schedule for a given number of inference steps
    fn set_timesteps(&mut self, num_steps: usize) -> Result<()>;

    /// Get the timesteps slice
    fn timesteps(&self) -> &[usize];

    /// Get the schedule sigmas slice
    fn sigmas(&self) -> &[f64];

    /// Initial noise standard deviation (e.g. sigma_max or 1.0)
    fn init_noise_sigma(&self) -> f64 {
        self.sigmas().first().copied().unwrap_or(1.0)
    }

    /// Predict the previous sample (x_{t-1}) from noise prediction and current sample (x_t)
    fn step(
        &mut self,
        model_output: &Tensor,
        timestep: usize,
        sample: &Tensor,
    ) -> Result<Tensor>;

    /// Scale model input according to current timestep requirements (e.g. Euler: sample / sqrt(sigma^2 + 1))
    fn scale_model_input(&self, sample: &Tensor, timestep: usize) -> Result<Tensor>;
}
