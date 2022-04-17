//! An error when modeling a Linear algorithm
use linfa::Float;
use thiserror::Error;

pub type Result<T, F> = std::result::Result<T, LinearError<F>>;

/// An error when modeling a Linear algorithm
#[derive(Error, Debug)]
pub enum LinearError<F: Float> {
    /// Errors encountered when using argmin's solver
    #[error("argmin {0}")]
    Argmin(#[from] argmin::core::Error),
    #[error(transparent)]
    BaseCrate(#[from] linfa::Error),
    #[error("At least one sample needed")]
    NotEnoughSamples,
    #[error("At least one target needed")]
    NotEnoughTargets,
    #[error("penalty should be positive, but is {0}")]
    InvalidPenalty(F),
    #[error("tweedie distribution power should not be in (0, 1), but is {0}")]
    InvalidTweediePower(F),
    #[error("some value(s) of y are out of the valid range for power value {0}")]
    InvalidTargetRange(F),
    #[error(transparent)]
    #[cfg(feature = "blas")]
    LinalgBlasError(#[from] ndarray_linalg::error::LinalgError),
    #[error(transparent)]
    LinalgError(#[from] ndarray_linalg_rs::LinalgError),
}
