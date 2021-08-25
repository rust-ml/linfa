//! Error types in Linfa
//!

use thiserror::Error;

use ndarray::ShapeError;
#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};

pub type Result<T> = std::result::Result<T, Error>;

#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Error, Debug, Clone)]
pub enum Error {
    #[error("invalid parameter {0}")]
    Parameters(String),
    #[error("invalid prior {0}")]
    Priors(String),
    #[error("algorithm not converged {0}")]
    NotConverged(String),
    // ShapeError doesn't implement serde traits, and deriving them remotely on a complex error
    // type isn't really feasible, so we skip this variant.
    #[cfg_attr(feature = "serde", serde(skip))]
    #[error("invalid ndarray shape {0}")]
    NdShape(#[from] ShapeError),
    #[error("not enough samples")]
    NotEnoughSamples,
    #[error("multiple targets not supported")]
    MultipleTargets,
    #[error("The number of samples do not match: {0} - {1}")]
    MismatchedShapes(usize, usize),
}
