// [WFGY] Zone: SAFE | λ: 0.15 | Fallbacks: 0 | Action: Lumina root library and API re-exports

pub mod device;
pub mod diffusion;
pub mod error;
pub mod pipelines;
pub mod text;
pub mod traits;
pub mod weights;

pub use device::{auto_device, select_device};
pub use error::{LuminaError, Result};
pub use traits::{DiffusionParams, TextGenerationPipeline, TextToImagePipeline};
pub use weights::{SafeTensorsArchive, WeightRouter};

pub use diffusion::schedulers::{
    DDIMConfig, DDIMScheduler, EulerDiscreteScheduler, EulerSchedulerConfig, PredictionType,
    Scheduler,
};
pub use diffusion::unet_2d::UNetConditionModel;
pub use diffusion::vae::{FastLatentPreviewer, VaeDecoder, tensor_to_rgb_image};
pub use pipelines::{StableDiffusionPipeline, StableDiffusionXLPipeline};
pub use text::{ClipTextEncoder, OpenClipTextEncoder};

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn test_euler_scheduler_standard_steps() {
        let config = EulerSchedulerConfig::default();
        let mut scheduler = EulerDiscreteScheduler::new(config);
        assert!(scheduler.set_timesteps(20).is_ok());
        assert_eq!(scheduler.timesteps().len(), 20);

        let device = Device::Cpu;
        let sample = Tensor::zeros((1, 4, 64, 64), candle_core::DType::F32, &device).unwrap();
        let model_output = Tensor::zeros((1, 4, 64, 64), candle_core::DType::F32, &device).unwrap();
        let t = scheduler.timesteps()[0];

        let scaled = scheduler.scale_model_input(&sample, t).unwrap();
        assert_eq!(scaled.shape().dims(), &[1, 4, 64, 64]);

        let next_sample = scheduler.step(&model_output, t, &sample).unwrap();
        assert_eq!(next_sample.shape().dims(), &[1, 4, 64, 64]);
    }

    #[test]
    fn test_euler_scheduler_karras_sigmas() {
        let mut config = EulerSchedulerConfig::default();
        config.use_karras_sigmas = true;
        let mut scheduler = EulerDiscreteScheduler::new(config);
        assert!(scheduler.set_timesteps(25).is_ok());
        assert_eq!(scheduler.timesteps().len(), 25);
        assert_eq!(scheduler.sigmas().len(), 26);

        // Verify monotonic decrease of Karras sigmas
        let sigmas = scheduler.sigmas();
        for i in 0..sigmas.len() - 1 {
            assert!(sigmas[i] >= sigmas[i + 1]);
        }
    }

    #[test]
    fn test_ddim_scheduler_steps() {
        let config = DDIMConfig::default();
        let mut scheduler = DDIMScheduler::new(config);
        assert!(scheduler.set_timesteps(10).is_ok());
        assert_eq!(scheduler.timesteps().len(), 10);

        let device = Device::Cpu;
        let sample = Tensor::zeros((1, 4, 64, 64), candle_core::DType::F32, &device).unwrap();
        let model_output = Tensor::zeros((1, 4, 64, 64), candle_core::DType::F32, &device).unwrap();
        let t = scheduler.timesteps()[0];

        let next_sample = scheduler.step(&model_output, t, &sample).unwrap();
        assert_eq!(next_sample.shape().dims(), &[1, 4, 64, 64]);
    }

    #[test]
    fn test_fast_latent_previewer() {
        let device = Device::Cpu;
        let dummy_latent = Tensor::randn(0.0f32, 1.0f32, (1, 4, 32, 32), &device).unwrap();
        let img = FastLatentPreviewer::preview_latent(&dummy_latent).unwrap();
        assert_eq!(img.width(), 32);
        assert_eq!(img.height(), 32);
    }

    #[test]
    fn test_tensor_to_rgb_image() {
        let device = Device::Cpu;
        let dummy_rgb = Tensor::zeros((1, 3, 64, 64), candle_core::DType::F32, &device).unwrap();
        let img = tensor_to_rgb_image(&dummy_rgb).unwrap();
        assert_eq!(img.width(), 64);
        assert_eq!(img.height(), 64);
    }

    #[test]
    fn test_diffusion_params_default() {
        let params = DiffusionParams::default();
        assert_eq!(params.num_steps, 25);
        assert_eq!(params.guidance_scale, 7.5);
        assert_eq!(params.width, 512);
        assert_eq!(params.height, 512);
    }
}
