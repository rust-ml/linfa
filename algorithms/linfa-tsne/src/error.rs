use thiserror::Error;

/// Simplified `Result` using [`TSneError`](crate::TSneError) as error type
pub type Result<T> = std::result::Result<T, TSneError>;

/// Error variants from hyper-parameter construction or model estimation
#[derive(Error, Debug, Clone)]
pub enum TSneError {
    #[error("negative perplexity")]
    NegativePerplexity,
    #[error("perplexity too large for number of samples")]
    PerplexityTooLarge,
    #[error("negative approximation threshold")]
    NegativeApproximationThreshold,
    #[error("embedding size larger than original dimensionality")]
    EmbeddingSizeTooLarge,
    #[error("number of preliminary iterations larger than total iterations")]
    PreliminaryIterationsTooLarge,
    #[error("invalid shaped array {0}")]
    InvalidShape(#[from] ndarray::ShapeError),
    #[error(transparent)]
    BaseCrate(#[from] linfa::Error),
}
