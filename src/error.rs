//! Error types in Linfa
//!

use std::error::Error as StdError;
use std::fmt;

use ndarray::ShapeError;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    Parameters(String),
    Priors(String),
    NotConverged(String),
    NdShape(ShapeError),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Parameters(msg) => write!(f, "Parameter: {}", msg),
            Error::NdShape(msg) => write!(f, "NdArray shape: {}", msg),
            Error::Priors(msg) => write!(f, "Priors: {}", msg),
            Error::NotConverged(msg) => write!(f, "Not converged: {}", msg)
        }
    }
}

impl StdError for Error {}
