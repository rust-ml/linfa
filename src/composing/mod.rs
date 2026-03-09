//! Composition models
//!
//! This module contains four composition models:
//!  * `MultiClassModel`: combine multiple binary decision models to a single multi-class model
//!  * `MultiTargetModel`: combine multiple univariate models to a single multi-target model
//!  * `Platt`: calibrate a classifier (i.e. SVC) to predicted posterior probabilities
//!  * `ResidualChain`: fit models sequentially on the residuals of the previous one
//!    (forward stagewise additive modeling / L2Boosting); see [`residual_chain::Stagewise`]
mod multi_class_model;
mod multi_target_model;
pub mod platt_scaling;
pub mod residual_chain;

pub use multi_class_model::MultiClassModel;
pub use multi_target_model::MultiTargetModel;
pub use platt_scaling::{Platt, PlattError, PlattParams};
