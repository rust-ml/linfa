//! Error definitions for preprocessing
use thiserror::Error;
pub type Result<T> = std::result::Result<T, PreprocessingError>;

#[derive(Error, Debug)]
#[non_exhaustive]
pub enum PreprocessingError {
    #[error("wrong measure ({0}) for scaler: {1}")]
    WrongMeasureForScaler(String, String),
    #[error("subsamples greater than total samples: {0} > {1}")]
    TooManySubsamples(usize, usize),
    #[error("not enough samples")]
    NotEnoughSamples,
    #[error("not a valid float")]
    InvalidFloat,
    #[error("minimum value for MinMax scaler cannot be greater than the maximum")]
    TokenizerNotSet,
    #[error("Tokenizer must be defined after deserializing CountVectorizer by calling force_tokenizer_redefinition")]
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
    #[error(transparent)]
    IoError(#[from] std::io::Error),
    #[error("Encoding error {0}")]
    EncodingError(std::borrow::Cow<'static, str>),
    #[cfg(feature = "blas")]
    #[error(transparent)]
    LinalgBlasError(#[from] ndarray_linalg::error::LinalgError),
    #[error(transparent)]
    LinalgError(#[from] linfa_linalg::LinalgError),
    #[error(transparent)]
    NdarrayStatsEmptyError(#[from] ndarray_stats::errors::EmptyInput),
    #[error(transparent)]
    LinfaError(#[from] linfa::error::Error),
}
