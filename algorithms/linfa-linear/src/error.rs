use std::error::Error;
use std::fmt::{self, Display};

pub type Result<T> = std::result::Result<T, LinearError>;

/// An error when modeling a Linear algorithm
#[derive(Debug)]
pub enum LinearError {
    /// When any of the hyperparameters are set the wrong value
    InvalidValue(String),
    /// Errors encountered when using argmin's solver
    Argmin(argmin::core::Error),
}

impl Display for LinearError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::InvalidValue(message) => write!(f, "Invalid value encountered: {}", message),
            Self::Argmin(error) => write!(f, "Argmin Error: {}", error),
        }
    }
}

impl Error for LinearError {}

impl From<argmin::core::Error> for LinearError {
    fn from(error: argmin::core::Error) -> LinearError {
        LinearError::Argmin(error)
    }
}
