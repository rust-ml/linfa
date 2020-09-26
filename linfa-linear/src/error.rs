use std::error::Error;
use std::fmt::{self, Display};

pub type Result<T> = std::result::Result<T, LinearError>;

#[derive(Debug)]
pub enum LinearError {
    InvalidValue(String),
    ArgminError(argmin::core::Error),
}

impl Display for LinearError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::InvalidValue(message) => write!(f, "Invalid value encountered: {}", message),
            Self::ArgminError(error) => write!(f, "Argmin Error: {}", error),
        }
    }
}

impl Error for LinearError {}

impl From<argmin::core::Error> for LinearError {
    fn from(error: argmin::core::Error) -> LinearError {
        LinearError::ArgminError(error)
    }
}
