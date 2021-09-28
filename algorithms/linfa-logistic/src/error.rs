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
    #[error("Rows of initial parameter ({rows}) must be the same as the number of features ({n_features})")]
    InitialParameterFeaturesMismatch { rows: usize, n_features: usize },
    #[error("Columns of initial parameter ({cols}) must be the same as the number of classes ({n_classes})")]
    InitialParameterClassesMismatch { cols: usize, n_classes: usize },

    #[error("gradient_tolerance must be a positive, finite number")]
    InvalidGradientTolerance,
    #[error("alpha must be a positive, finite number")]
    InvalidAlpha,
    #[error("Initial parameters must be finite")]
    InvalidInitialParameters,
}
