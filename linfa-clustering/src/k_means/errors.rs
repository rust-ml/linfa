use std::error::Error;
use std::fmt::{self, Display};

pub type Result<T> = std::result::Result<T, KMeansError>;

/// An error when modeling a KMeans algorithm
#[derive(Debug)]
pub enum KMeansError {
    /// When any of the hyperparameters are set the wrong value
    InvalidValue(String),
    /// When inertia computation fails
    InertiaError(String),
    /// When fitting algorithm does not converge
    NotConverged(String),
}

impl Display for KMeansError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::InvalidValue(message) => write!(f, "Invalid value encountered: {}", message),
            Self::InertiaError(message) => write!(f, "Fitting failed: {}", message),
            Self::NotConverged(message) => write!(f, "Fitting failed: {}", message),
        }
    }
}

impl Error for KMeansError {}
