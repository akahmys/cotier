#![forbid(unsafe_code)]

pub mod engine;
pub mod learner;
pub mod memory;
pub mod model;
pub mod server;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum CotierError {
    #[error("Tensor error: {0}")]
    Tensor(#[from] candle_core::Error),

    #[error("Database error: {0}")]
    Database(#[from] rusqlite::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Model load error: {0}")]
    ModelLoad(String),

    #[error("Inference error: {0}")]
    Inference(String),
}

pub type Result<T> = std::result::Result<T, CotierError>;
