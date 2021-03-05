//! Error types in Linfa
//!

use thiserror::Error;

use ndarray::ShapeError;
pub type Result<T> = std::result::Result<T, Error>;

#[derive(Error, Debug, Clone)]
pub enum Error {
    #[error("invalid parameter {0}")]
    Parameters(String),
    #[error("invalid prior {0}")]
    Priors(String),
    #[error("algorithm not converged {0}")]
    NotConverged(String),
    #[error("invalid ndarray shape {0}")]
    NdShape(#[from] ShapeError),
    #[error("multiple targets not supported")]
    MultipleTargets,
    #[error("Not enough samples to compute the mean")]
    NotEnoughSamples,
}
