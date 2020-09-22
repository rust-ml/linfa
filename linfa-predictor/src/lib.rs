use ndarray::{Array1, ArrayBase, Data, Ix2};
use std::error::Error;
use std::fmt;

#[derive(Debug)]
pub struct LinfaError {
    details: String,
}

impl LinfaError {
    pub fn new(msg: &str) -> LinfaError {
        LinfaError {
            details: msg.to_string(),
        }
    }
}

impl fmt::Display for LinfaError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.details)
    }
}

impl Error for LinfaError {
    fn description(&self) -> &str {
        &self.details
    }
}

/// Trait every predictor should implement
pub trait Predictor {
    /// predict class for each sample
    fn predict(&self, x: &ArrayBase<impl Data<Elem = f64>, Ix2>)
        -> Result<Array1<u64>, LinfaError>;

    /// predict probability of each possible class for each sample
    fn predict_probabilities(
        &self,
        x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Result<Vec<Array1<f64>>, LinfaError>;
}
