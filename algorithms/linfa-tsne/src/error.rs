use thiserror::Error;

pub type Result<T> = std::result::Result<T, TSneError>;

#[derive(Error, Debug)]
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
