use ndarray_linalg::error::LinalgError;
use thiserror::Error;

pub type Result<T> = std::result::Result<T, FastIcaError>;

/// An error when modeling FastICA algorithm
#[derive(Error, Debug)]
pub enum FastIcaError {
    /// When there are no samples in the provided dataset
    #[error("Dataset must contain at least one sample")]
    NotEnoughSamples,
    /// When any of the hyperparameters are set the wrong value
    #[error("Invalid value encountered: {0}")]
    InvalidValue(String),
    /// If we fail to compute any components of the SVD decomposition
    /// due to an Ill-Conditioned matrix
    #[error("SVD Decomposition failed, X could be an Ill-Conditioned matrix")]
    SvdDecomposition,
    #[error("tolerance should be positive but is {0}")]
    InvalidTolerance(f32),
    /// Errors encountered during linear algebra operations
    #[error("Linalg Error: {0}")]
    Linalg(#[from] LinalgError),
    #[error(transparent)]
    LinfaError(#[from] linfa::error::Error),
}
