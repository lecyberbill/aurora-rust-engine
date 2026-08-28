// [WFGY] Zone: SAFE | λ: 0.15 | Fallbacks: 0 | Action: Engine error taxonomy and context propagation

use thiserror::Error;

#[derive(Error, Debug)]
pub enum LuminaError {
    #[error("Candle tensor operation error: {0}")]
    Candle(#[from] candle_core::Error),

    #[error("Tokenizer error: {0}")]
    Tokenizer(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("SafeTensors parsing error: {0}")]
    SafeTensor(#[from] safetensors::SafeTensorError),

    #[error("Model configuration error: {0}")]
    Config(String),

    #[error("Weight tensor missing or mismatched: {0}")]
    MissingWeight(String),

    #[error("Dimension mismatch: expected {expected:?}, got {actual:?}")]
    DimensionMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("Scheduler invariant violation: {0}")]
    SchedulerInvariant(String),

    #[error("Pipeline error: {0}")]
    Pipeline(String),

    #[error("{context}")]
    Context { context: String, #[source] source: Box<LuminaError> },
}

pub type Result<T> = std::result::Result<T, LuminaError>;
