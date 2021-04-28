use thiserror::Error;

pub type Result<T> = std::result::Result<T, KMeansError>;

/// An error when modeling a KMeans algorithm
#[derive(Error, Debug)]
pub enum KMeansError {
    /// When any of the hyperparameters are set the wrong value
    #[error("Invalid value encountered: {0}")]
    InvalidValue(String),
    /// When inertia computation fails
    #[error("Fitting failed: {0}")]
    InertiaError(String),
    /// When fitting algorithm does not converge
    #[error("Fitting failed: {0}")]
    NotConverged(String),
    #[error(transparent)]
    LinfaError(#[from] linfa::error::Error),
}
