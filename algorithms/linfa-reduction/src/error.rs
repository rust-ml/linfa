use thiserror::Error;
pub type Result<T> = std::result::Result<T, Error>;

#[derive(Error, Debug)]
pub enum Error {
    #[error("At least 1 sample needed")]
    NotEnoughSamples,
    #[error(transparent)]
    LinalgError(#[from] ndarray_linalg::error::LinalgError),
    #[error(transparent)]
    LinfaError(#[from] linfa::error::Error),
}
