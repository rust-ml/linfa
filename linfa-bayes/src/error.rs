use std::fmt;

use ndarray::ShapeError;
use ndarray_stats::errors::MinMaxError;

pub type Result<T> = std::result::Result<T, BayesError>;

#[derive(Debug)]
pub enum BayesError {
    /// Error when accessing an untrained model
    UntrainedModel(String),
    /// Error when wrong input is sent
    Input(String),
    /// Error when the user sets a wrong Class prior
    Priors(String),
    /// Error when performing Max operation on data
    Stats(MinMaxError),
    /// Error when constructing an Array with the wrong shape
    Shape(ShapeError),
}

impl fmt::Display for BayesError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::UntrainedModel(msg) => write!(f, "{}", msg),
            Self::Input(msg) => write!(f, "{}", msg),
            Self::Priors(msg) => write!(f, "{}", msg),
            Self::Stats(error) => write!(f, "Ndarray Stats Error: {}", error),
            Self::Shape(error) => write!(f, "Ndarray Shape Error: {}", error),
        }
    }
}

impl From<MinMaxError> for BayesError {
    fn from(error: MinMaxError) -> Self {
        Self::Stats(error)
    }
}

impl From<ShapeError> for BayesError {
    fn from(error: ShapeError) -> Self {
        Self::Shape(error)
    }
}

impl std::error::Error for BayesError {}
