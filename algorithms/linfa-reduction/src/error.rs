use thiserror::Error;

pub type Result<T> = std::result::Result<T, ReductionError>;

#[derive(Error, Debug)]
#[non_exhaustive]
pub enum ReductionError {
    #[error("At least 1 sample needed")]
    NotEnoughSamples,
    #[error("embedding dimension smaller {0} than feature dimension")]
    EmbeddingTooSmall(usize),
    #[error("Number of steps zero in diffusion map operator")]
    StepsZero,
    #[cfg(feature = "blas")]
    #[error(transparent)]
    LinalgBlasError(#[from] ndarray_linalg::error::LinalgError),
    #[error(transparent)]
    LinalgError(#[from] linfa_linalg::LinalgError),
    #[error(transparent)]
    LinfaError(#[from] linfa::error::Error),
    #[error(transparent)]
    NdarrayRandError(#[from] ndarray_rand::rand_distr::NormalError),
    #[error("Precision parameter must be in the interval (0; 1)")]
    InvalidPrecision,
    #[error("Target dimension of the projection must be positive")]
    NonPositiveEmbeddingSize,
    #[error("Target dimension {0} is larger than the number of features {1}.")]
    DimensionIncrease(usize, usize),
}
