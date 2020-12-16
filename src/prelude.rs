//! Linfa prelude.
//!
//! This module contains the most used types, type aliases, traits and
//! functions that you can import easily as a group.
//!

#[doc(no_inline)]
pub use crate::error::{Error, Result};

#[doc(no_inline)]
pub use crate::traits::*;

#[doc(no_inline)]
pub use crate::dataset::{Dataset, Float, Labels, Records, Targets};

#[doc(no_inline)]
pub use crate::metrics_classification::{BinaryClassification, ConfusionMatrix, ToConfusionMatrix};

#[doc(no_inline)]
pub use crate::metrics_regression::Regression;
