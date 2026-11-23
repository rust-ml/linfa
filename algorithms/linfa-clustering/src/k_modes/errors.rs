use thiserror::Error;

#[derive(Error, Debug, PartialEq)]
pub enum KModesParamsError {
    #[error("n_clusters cannot be 0")]
    NClusters,
    #[error("max_n_iterations cannot be 0")]
    MaxIterations,
    #[error("n_runs cannot be 0")]
    NRuns,
}

#[derive(Error, Debug)]
pub enum KModesError {
    #[error("Invalid hyperparameter: {0}")]
    InvalidParams(#[from] KModesParamsError),
    #[error(transparent)]
    LinfaError(#[from] linfa::error::Error),
}
