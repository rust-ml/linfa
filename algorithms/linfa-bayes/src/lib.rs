#![doc = include_str!("../README.md")]

mod error;
mod gaussian_nb;
mod hyperparams;

pub use error::{NaiveBayesError, Result};
pub use gaussian_nb::GaussianNb;
pub use hyperparams::{GaussianNbParams, GaussianNbValidParams};
