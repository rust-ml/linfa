use thiserror::Error;
pub type Result<T> = std::result::Result<T, Error>;

#[derive(Error, Debug, Clone)]
pub enum Error {
    #[error("wrong measure ({0}) for scaler: {1}")]
    WrongMeasureForScaler(String, String),
    #[error("subsamples greater than total samples: {0} > {1}")]
    TooManySubsamples(usize, usize),
    #[error("not enough samples")]
    NotEnoughSamples,
    #[error("not a valid float")]
    InvalidFloat,
}
