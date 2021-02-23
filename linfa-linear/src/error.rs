use thiserror::Error;

pub type Result<T> = std::result::Result<T, LinearError>;

/// An error when modeling a Linear algorithm
#[derive(Error, Debug)]
pub enum LinearError {
    /// Errors encountered when using argmin's solver
    #[error("argmin {0}")]
    Argmin(#[from] argmin::core::Error),
    #[error(transparent)]
    BaseCrate(#[from] linfa::Error),
}
