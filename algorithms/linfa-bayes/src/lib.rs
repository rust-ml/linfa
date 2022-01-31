#![doc = include_str!("../README.md")]

mod base_nb;
mod error;
mod gaussian_nb;
mod hyperparams;
mod multinomial_nb;

pub use error::{NaiveBayesError, Result};
pub use gaussian_nb::GaussianNb;
pub use hyperparams::{GaussianNbParams, GaussianNbValidParams};
pub use hyperparams::{MultinomialNbParams, MultinomialNbValidParams};
pub use multinomial_nb::MultinomialNb;
