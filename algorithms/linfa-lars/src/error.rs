use thiserror::Error;

#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Error,Debug,Clone)]
pub enum LarsError{
    #[error(transparent)]
    BaseCrate(#[from] linfa::Error),
    #[error("invalid epsilon")]
    InvalidEpsilon
}