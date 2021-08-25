use thiserror::Error;
pub type Result<T> = std::result::Result<T, ReductionError>;

#[derive(Error, Debug)]
pub enum ReductionError {
    #[error("At least 1 sample needed")]
    NotEnoughSamples,
    #[error("embedding dimension smaller {0} than feature dimension")]
    EmbeddingTooSmall(usize),
    #[error("Number of steps zero in diffusion map operator")]
    StepsZero,
    #[error(transparent)]
    LinalgError(#[from] ndarray_linalg::error::LinalgError),
    #[error(transparent)]
    LinfaError(#[from] linfa::error::Error),
}
