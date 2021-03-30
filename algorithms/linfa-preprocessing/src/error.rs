//! Error definitions for preprocessing
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
    #[error("minimum value for MinMax scaler cannot be greater than the maximum")]
    FlippedMinMaxRange,
    #[error("n_gram boundaries cannot be zero (min = {0}, max = {1})")]
    InvalidNGramBoundaries(usize, usize),
    #[error("n_gram min boundary cannot be greater than max boundary (min = {0}, max = {1})")]
    FlippedNGramBoundaries(usize, usize),
    #[error("document frequencies have to be between 0 and 1 (min = {0}, max = {1})")]
    InvalidDocumentFrequencies(f32, f32),
    #[error("min document frequency cannot be greater than max document frequency (min = {0}, max = {1})")]
    FlippedDocumentFrequencies(f32, f32),
    #[error(transparent)]
    RegexError(#[from] regex::Error),
}
