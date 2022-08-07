use thiserror::Error;

pub type Result<T> = std::result::Result<T, FastIcaError>;

/// An error when modeling FastICA algorithm
#[derive(Error, Debug)]
#[non_exhaustive]
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
    #[cfg(feature = "blas")]
    #[error("Linalg BLAS error: {0}")]
    LinalgBlasError(#[from] ndarray_linalg::error::LinalgError),
    #[error("Linalg error: {0}")]
    /// Errors encountered during linear algebra operations
    LinalgError(#[from] linfa_linalg::LinalgError),
    #[error(transparent)]
    LinfaError(#[from] linfa::error::Error),
}
