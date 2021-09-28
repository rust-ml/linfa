//! Composition models
//!
//! This module contains three composition models:
//!  * `MultiClassModel`: combine multiple binary decision models to a single multi-class model
//!  * `MultiTargetModel`: combine multiple univariate models to a single multi-target model
//!  * `Platt`: calibrate a classifier (i.e. SVC) to predicted posterior probabilities
mod multi_class_model;
mod multi_target_model;
pub mod platt_scaling;

pub use multi_class_model::MultiClassModel;
pub use multi_target_model::MultiTargetModel;
pub use platt_scaling::{Platt, PlattError, PlattParams};
