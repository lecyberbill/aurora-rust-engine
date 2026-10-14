// [WFGY] Zone: SAFE | λ: 0.15 | Fallbacks: 0 | Action: Diffusion subsystem re-exports

pub mod attention;
pub mod controlnet;
pub mod controlnet_mmdit;
pub mod dit;
pub mod ip_adapter;
pub mod schedulers;
pub mod unet_2d;
pub mod vae;
pub mod vae_flux;
pub mod vae_qwen;
pub mod vae_video;
pub mod vae_minimax_audio;
pub mod vae_minimax_video;
pub mod video_rope;

pub use attention::{CrossAttention, SpatialTransformer};
pub use controlnet::{ControlNetModel, MultiControlNet, compute_canny_edge_map};
pub use controlnet_mmdit::{MMDiTControlNetAdapter, MMDiTControlNetManager};
pub use dit::{DiffusionTransformer, PixArtConfig, PixArtTransformer};
pub use ip_adapter::{IPAdapter, IPAdapterCrossAttention, ImageProjModel};
pub use schedulers::{DDIMScheduler, EulerDiscreteScheduler, FlowMatchEulerScheduler, Scheduler};
pub use unet_2d::UNetConditionModel;
pub use vae::{FastLatentPreviewer, VaeDecoder, tensor_to_rgb_image};
pub use vae_flux::{FluxVaeDecoder, FluxVaeEncoder};
pub use vae_minimax_audio::MiniMaxH3AudioVaeDecoder;
pub use vae_minimax_video::MiniMaxH3VideoVaeDecoder;
pub use vae_qwen::QwenImageVaeDecoder;
pub use vae_video::{Causal3DVideoVaeDecoder, VideoVaeConfig};
pub use video_rope::{VideoGenerationParams, VideoRope3D};
