use ndarray_linalg::error::LinalgError;
use std::error::Error;
use std::fmt::{self, Display};

pub type Result<T> = std::result::Result<T, PlsError>;

#[derive(Debug)]
pub enum PlsError {
    NotEnoughSamplesError(String),
    BadComponentNumberError(String),
    PowerMethodNotConvergedError(String),
    LinalgError(String),
}

impl Display for PlsError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::NotEnoughSamplesError(message) => {
                write!(f, "Not enough samples: {}", message)
            }
            Self::BadComponentNumberError(message) => {
                write!(f, "Bad component number: {}", message)
            }
            Self::PowerMethodNotConvergedError(message) => {
                write!(f, "Power method not converged: {}", message)
            }
            Self::LinalgError(message) => {
                write!(f, "Linear algebra error: {}", message)
            }
        }
    }
}

impl Error for PlsError {}

impl From<LinalgError> for PlsError {
    fn from(error: LinalgError) -> PlsError {
        PlsError::LinalgError(error.to_string())
    }
}
