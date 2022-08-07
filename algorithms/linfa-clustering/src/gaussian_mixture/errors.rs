use crate::k_means::KMeansError;
#[cfg(not(feature = "blas"))]
use linfa_linalg::LinalgError;
#[cfg(feature = "blas")]
use ndarray_linalg::error::LinalgError;
use thiserror::Error;
pub type Result<T> = std::result::Result<T, GmmError>;

/// An error when modeling a GMM algorithm
#[derive(Error, Debug)]
pub enum GmmError {
    /// When any of the hyperparameters are set the wrong value
    #[error("Invalid value encountered: {0}")]
    InvalidValue(String),
    /// Errors encountered during linear algebra operations
    #[error(
        "Linalg Error: \
    Fitting the mixture model failed because some components have \
    ill-defined empirical covariance (for instance caused by singleton \
    or collapsed samples). Try to decrease the number of components, \
    or increase reg_covar. Error: {0}"
    )]
    LinalgError(#[from] LinalgError),
    /// When a cluster has no more data point while fitting GMM
    #[error("Fitting failed: {0}")]
    EmptyCluster(String),
    /// When lower bound computation fails
    #[error("Fitting failed: {0}")]
    LowerBoundError(String),
    /// When fitting EM algorithm does not converge
    #[error("Fitting failed: {0}")]
    NotConverged(String),
    /// When initial KMeans fails
    #[error("Initial KMeans failed: {0}")]
    KMeansError(#[from] KMeansError),
    #[error(transparent)]
    LinfaError(#[from] linfa::error::Error),
    #[error(transparent)]
    MinMaxError(#[from] ndarray_stats::errors::MinMaxError),
}
