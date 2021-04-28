//! Error types in Linfa
//!

use thiserror::Error;

use crate::composing::PlattNewtonResult;
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
    #[error("not enough samples")]
    NotEnoughSamples,
    #[error("multiple targets not supported")]
    MultipleTargets,
    #[error("platt scaling failed")]
    Platt(PlattNewtonResult),
    #[error("The number of samples do not match: {0} - {1}")]
    MismatchedShapes(usize, usize),
}
