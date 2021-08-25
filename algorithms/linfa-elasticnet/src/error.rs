#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};
use thiserror::Error;

/// Simplified `Result` using [`ElasticNetError`](crate::ElasticNetError) as error type
pub type Result<T> = std::result::Result<T, ElasticNetError>;

#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
/// Error variants from hyperparameter construction or model estimation
#[derive(Debug, Clone, Error)]
pub enum ElasticNetError {
    /// The input has not enough samples
    #[error("not enough samples as they have to be larger than number of features")]
    NotEnoughSamples,
    /// The input is singular
    #[error("the data is ill-conditioned")]
    IllConditioned,
    #[error("l1 ratio should be in range [0, 1], but is {0}")]
    InvalidL1Ratio(f32),
    #[error("invalid penalty {0}")]
    InvalidPenalty(f32),
    #[error("invalid tolerance {0}")]
    InvalidTolerance(f32),
    #[error(transparent)]
    BaseCrate(#[from] linfa::Error),
}
