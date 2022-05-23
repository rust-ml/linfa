//! Error definitions
//!

use crate::{Criterion, Float};
use thiserror::Error;

/// Simplified `Result` using [`HierarchicalError`](crate::HierarchicalError) as error type
pub type Result<T, F> = std::result::Result<T, HierarchicalError<F>>;

/// Error variants from parameter construction
#[derive(Error, Debug, Clone)]
pub enum HierarchicalError<F: Float> {
    /// Invalid stopping condition
    #[error("The stopping condition {0:?} is not valid")]
    InvalidStoppingCondition(Criterion<F>),
    #[error(transparent)]
    BaseCrate(#[from] linfa::Error),
}
