use ndarray_stats::errors::MinMaxError;
use thiserror::Error;

/// Simplified `Result` using [`NaiveBayesError`](crate::NaiveBayesError) as error type
pub type Result<T> = std::result::Result<T, NaiveBayesError>;

/// Error variants from hyper-parameter construction or model estimation
#[derive(Error, Debug, Clone)]
pub enum NaiveBayesError {
    /// Error when performing Max operation on data
    #[error("invalid statistical operation {0}")]
    Stats(#[from] MinMaxError),
    /// Invalid smoothing parameter
    #[error("invalid smoothing parameter {0}")]
    InvalidSmoothing(f64),
    #[error(transparent)]
    BaseCrate(#[from] linfa::Error),
}
