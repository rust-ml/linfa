use std::fmt;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    /// Indicate mis-configured hyper parameters
    InvalidParams(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::InvalidParams(msg) => write!(f, "Invalid hyper parameter: {}", msg),
        }
    }
}

/// Derive the std error type
impl std::error::Error for Error {}
