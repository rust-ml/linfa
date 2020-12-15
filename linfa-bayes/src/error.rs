use std::fmt;

use ndarray_stats::errors::MinMaxError;

pub type Result<T> = std::result::Result<T, BayesError>;

#[derive(Debug)]
pub enum BayesError {
    /// Error when performing Max operation on data
    Stats(MinMaxError),
}

impl fmt::Display for BayesError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Stats(error) => write!(f, "Ndarray Stats Error: {}", error),
        }
    }
}

impl From<MinMaxError> for BayesError {
    fn from(error: MinMaxError) -> Self {
        Self::Stats(error)
    }
}

impl std::error::Error for BayesError {}
