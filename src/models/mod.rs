// [WFGY] Zone: SAFE | λ: 0.25 | Fallbacks: 0 | Action: `transformers`-style model facade for Candle (Auto* + traits)

//! Port of the HuggingFace `transformers` API surface onto Candle.
//!
//! Today the engine ships several concrete bricks (Flux/SDXL pipelines, Qwen/Mistral/T5 text
//! encoders). This module **assembles** them behind one coherent `AutoModel::from_pretrained`
//! entry point plus a set of object-safe traits, so a platform writes a few lines to load and
//! infer any supported model instead of wiring pipelines by hand.

pub mod auto;
pub mod common;
pub mod config;
pub mod descriptor;
pub mod image;
pub mod registry;
pub mod text;

pub use auto::{AutoModel, BoxedModel, ModelLoadConfig};
pub use common::{downcast_model, AnyModel, EncodeModel, GenerationModel, ImageGenerationModel, ModelKind};
pub use config::{detect_architecture, Architecture};
pub use descriptor::{
    ModelDescriptor, ModelDescriptorEntry, ModelDescriptorFile, TextEncoderConfig, TextEncoderSpec,
};
pub use image::DiffusionModel;
pub use registry::ModelRegistry;
pub use text::TextModel;
