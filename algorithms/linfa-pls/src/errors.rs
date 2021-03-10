use std::error::Error;
use std::fmt::{self, Display};

pub type Result<T> = std::result::Result<T, PlsError>;

#[derive(Debug)]
pub struct PlsError {
    message: String,
}

impl PlsError {
    pub fn new(message: String) -> Self {
        PlsError { message }
    }
}

impl Display for PlsError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "PLS error: {}", self.message)
    }
}

impl Error for PlsError {}
