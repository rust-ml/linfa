use std::error::Error;
use std::fmt::{self, Display};

pub type Result<T> = std::result::Result<T, LinearError>;

#[derive(Debug)]
pub enum LinearError {
    InvalidValue(String),
}

impl Display for LinearError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::InvalidValue(message) => write!(f, "Invalid value encountered: {}", message),
        }
    }
}

impl Error for LinearError {}
