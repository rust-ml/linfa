use ndarray_linalg::error::LinalgError;
use thiserror::Error;
pub type Result<T> = std::result::Result<T, PlsError>;

#[derive(Error, Debug)]
pub enum PlsError {
    #[error("Not enough samples: {0}")]
    NotEnoughSamplesError(String),
    #[error("Bad component number: {0}")]
    BadComponentNumberError(String),
    #[error("The tolerance is should not be negative, NaN or inf but is {0}")]
    InvalidTolerance(f32),
    #[error("The maximal number of iterations should be positive")]
    ZeroMaxIter,
    #[error("Power method not converged: {0}")]
    PowerMethodNotConvergedError(String),
    #[error(transparent)]
    LinalgError(#[from] LinalgError),
    #[error(transparent)]
    LinfaError(#[from] linfa::error::Error),
    #[error(transparent)]
    MinMaxError(#[from] ndarray_stats::errors::MinMaxError),
}
