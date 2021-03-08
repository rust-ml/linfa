use thiserror::Error;
pub type Result<T> = std::result::Result<T, Error>;

#[derive(Error, Debug, Clone)]
pub enum Error {
    #[error("Wrong measure ({0}) for scaler: {1}")]
    WrongMeasureForScaler(String, String),
}
