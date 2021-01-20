use std::fmt;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone)]
pub enum Error {
    /// Indicate mis-configured hyper parameters
    InvalidParams(String),
    /// The input has not enough samples
    NotEnoughSamples,
    /// The input is singular
    IllConditioned,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::InvalidParams(msg) => write!(f, "Invalid hyper parameter: {}", msg),
            Self::NotEnoughSamples => write!(
                f,
                "Not enough samples, has to be larger than number of features"
            ),
            Self::IllConditioned => write!(f, "Ill conditioned data matrix"),
        }
    }
}

/// Derive the std error type
impl std::error::Error for Error {}
