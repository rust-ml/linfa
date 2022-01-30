#![doc = include_str!("../README.md")]

mod error;
mod base_nb;
mod gaussian_nb;
mod multinomial_nb;
mod hyperparams;

pub use error::{NaiveBayesError, Result};
pub use gaussian_nb::GaussianNb;
pub use hyperparams::{GaussianNbParams, GaussianNbValidParams};
pub use multinomial_nb::MultinomialNb;
pub use hyperparams::{MultinomialNbParams, MultinomialNbValidParams};

