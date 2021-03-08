#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};
use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Debug, Clone, Error)]
pub enum Error {
    /// The input has not enough samples
    #[error("not enough samples as they have to be larger than number of features")]
    NotEnoughSamples,
    /// The input is singular
    #[error("the data is ill-conditioned")]
    IllConditioned,
    #[error(transparent)]
    BaseCrate(#[from] linfa::Error),
}
