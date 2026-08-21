// [WFGY] Zone: SAFE | λ: 0.20 | Fallbacks: 0 | Action: Diffusion scheduler trait definition and factory

use candle_core::{Result, Tensor};

pub mod euler;
pub mod ddim;

pub use euler::{EulerDiscreteScheduler, EulerSchedulerConfig, PredictionType};
pub use ddim::{DDIMScheduler, DDIMConfig};

/// Mathematical scheduler contract for stepping iterative noise diffusion models.
pub trait Scheduler {
    /// Initialize timesteps schedule for a given number of inference steps
    fn set_timesteps(&mut self, num_steps: usize) -> Result<()>;

    /// Get the timesteps slice
    fn timesteps(&self) -> &[usize];

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
