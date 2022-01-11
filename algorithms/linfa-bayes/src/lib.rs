#![doc = include_str!("../README.md")]

mod error;
mod base_nb;
mod gaussian_nb;
mod multinomial_nb;
mod hyperparams;

pub use error::{NaiveBayesError, Result};
pub use base_nb::BaseNb;
pub use gaussian_nb::GaussianNb;
pub use multinomial_nb::MultinomialNb;
pub use hyperparams::{NbParams, NbValidParams};
