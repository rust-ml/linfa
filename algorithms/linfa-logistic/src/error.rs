use thiserror::Error;
pub type Result<T> = std::result::Result<T, Error>;

#[derive(Error, Debug)]
pub enum Error {
    #[error(transparent)]
    LinfaError(#[from] linfa::Error),
    #[error("Expected exactly two classes for logistic regression")]
    WrongNumberOfClasses,
    #[error(transparent)]
    ArgMinError(#[from] argmin::core::Error),
    #[error("Expected `x` and `y` to have same number of rows, got {0} != {1}")]
    MismatchedShapes(usize, usize),
    #[error("Values must be finite and not `Inf`, `-Inf` or `NaN`")]
    InvalidValues,
    #[error("gradient_tolerance must be a positive, finite number")]
    InvalidGradientTolerance,
    #[error("Size of initial parameter guess must be the same as the number of columns in the feature matrix `x`")]
    InvalidInitialParametersGuessSize,
    #[error("Initial parameter guess must be finite")]
    InvalidInitialParametersGuess,
}
