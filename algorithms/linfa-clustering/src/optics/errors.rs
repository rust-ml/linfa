use thiserror::Error;
pub type Result<T> = std::result::Result<T, OpticsError>;

/// An error when performing OPTICS Analysis
#[derive(Error, Debug, Clone, PartialEq, Eq, Hash)]
pub enum OpticsError {
    /// When any of the hyperparameters are set the wrong value
    #[error("Invalid value encountered: {0}")]
    InvalidValue(String),
}
