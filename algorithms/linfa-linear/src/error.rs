//! An error when modeling a Linear algorithm
use thiserror::Error;

pub type Result<T> = std::result::Result<T, LinearError>;

/// An error when modeling a Linear algorithm
#[derive(Error, Debug)]
pub enum LinearError {
    /// Errors encountered when using argmin's solver
    #[error("argmin {0}")]
    Argmin(#[from] argmin::core::Error),
    #[error(transparent)]
    BaseCrate(#[from] linfa::Error),
    #[error("At least one sample needed")]
    NotEnoughSamples,
    #[error("At least one target needed")]
    NotEnoughTargets,
    #[error(transparent)]
    LinalgError(#[from] ndarray_linalg::error::LinalgError),
}
