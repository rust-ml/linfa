#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};
use thiserror::Error;

#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Error, Debug, Clone)]
pub enum FtrlError {
    #[error("l1 ratio should be in range [0, 1], but is {0}")]
    InvalidL1Ratio(f32),
    #[error("l2 ratio should be in range [0, 1], but is {0}")]
    InvalidL2Ratio(f32),
    #[error("alpha should be positive and finite, but is {0}")]
    InvalidAlpha(f32),
    #[error("beta should be positive and finite, but is {0}")]
    InvalidBeta(f32),
    #[error("number of features must be bigger than 0, but is {0}")]
    InvalidNFeatures(usize),
    #[error(transparent)]
    LinfaError(#[from] linfa::error::Error),
}
