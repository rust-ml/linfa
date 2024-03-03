use thiserror::Error;

/// An error when performing OPTICS Analysis
#[derive(Error, Debug)]
pub enum OpticsError {
    /// When any of the hyperparameters are set the wrong value
    #[error("Invalid value encountered: {0}")]
    InvalidValue(String),
}
