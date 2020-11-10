use ndarray_linalg::error::LinalgError;
use std::error::Error;
use std::fmt::{self, Display};

pub type Result<T> = std::result::Result<T, GmmError>;

/// An error when modeling a GMM algorithm
#[derive(Debug)]
pub enum GmmError {
    /// When any of the hyperparameters are set the wrong value
    InvalidValue(String),
    /// Errors encountered during linear algebra operations
    LinalgError(LinalgError),
    /// When a cluster has no more data point while fitting GMM
    EmptyCluster(String),
    /// When lower bound computation fail
    LowerBoundError(String),
    /// When fitting EM algorithm does not converge
    NotConverged(String),
}

impl Display for GmmError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::InvalidValue(message) => write!(f, "Invalid value encountered: {}", message),
            Self::LinalgError(error) => write!(
                f,
                "Linalg Error: \
            Fitting the mixture model failed because some components have \
            ill-defined empirical covariance (for instance caused by singleton \
            or collapsed samples). Try to decrease the number of components, \
            or increase reg_covar. Error: {}",
                error
            ),
            Self::EmptyCluster(message) => write!(f, "Empty cluster: {}", message),
        }
    }
}

impl Error for GmmError {}

impl From<LinalgError> for GmmError {
    fn from(error: LinalgError) -> GmmError {
        GmmError::LinalgError(error)
    }
}
