use ndarray_linalg::error::LinalgError;
use std::error::Error;
use std::fmt::{self, Display};

pub type Result<T> = std::result::Result<T, FastIcaError>;

/// An error when modeling FastICA algorithm 
#[derive(Debug)]
pub enum FastIcaError {
    /// When any of the hyperparameters are set the wrong value
    InvalidValue(String),
    /// If we fail to compute any components of the SVD decomposition
    /// due to an Ill-Conditioned matrix
    SvdDecomposition,
    /// Errors encountered during linear algebra operations
    Linalg(LinalgError),
}

impl Display for FastIcaError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::InvalidValue(message) => write!(f, "Invalid value encountered: {}", message),
            Self::SvdDecomposition => write!(
                f,
                "SVD Decomposition failed, X could be an Ill-Conditioned matrix",
            ),
            Self::Linalg(error) => write!(f, "Linalg Error: {}", error),
        }
    }
}

impl Error for FastIcaError {}

impl From<LinalgError> for FastIcaError {
    fn from(error: LinalgError) -> FastIcaError {
        FastIcaError::Linalg(error)
    }
}
