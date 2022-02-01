use ndarray_linalg::error::LinalgError;
use thiserror::Error;
pub type Result<T> = std::result::Result<T, PlsError>;

#[derive(Error, Debug)]
pub enum PlsError {
    #[error("Number of samples should be greater than 1, got {0}")]
    NotEnoughSamplesError(usize),
    #[error("Number of components should be in [1, {upperbound}], got {actual}")]
    BadComponentNumberError { upperbound: usize, actual: usize },
    #[error("The tolerance is should not be negative, NaN or inf but is {0}")]
    InvalidTolerance(f32),
    #[error("The maximal number of iterations should be positive")]
    ZeroMaxIter,
    #[error("Singular vector computation power method: max iterations ({0}) reached")]
    PowerMethodNotConvergedError(usize),
    #[error("Constant residual detected in power method")]
    PowerMethodConstantResidualError(),
    #[error(transparent)]
    LinalgError(#[from] LinalgError),
    #[error(transparent)]
    LinfaError(#[from] linfa::error::Error),
    #[error(transparent)]
    MinMaxError(#[from] ndarray_stats::errors::MinMaxError),
}
