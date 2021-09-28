use thiserror::Error;

pub type Result<T> = std::result::Result<T, SvmError>;

#[derive(Error, Debug)]
pub enum SvmError {
    #[error("Invalid epsilon {0}")]
    InvalidEps(f32),
    #[error("Negative C value {0:?} (positive, negative samples")]
    InvalidC((f32, f32)),
    #[error("Nu should be in unit range, is {0}")]
    InvalidNu(f32),
    #[error("platt scaling failed")]
    Platt(#[from] linfa::composing::PlattError),
    #[error(transparent)]
    BaseCrate(#[from] linfa::Error),
}
