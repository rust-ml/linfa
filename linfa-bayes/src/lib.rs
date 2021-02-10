//! # Naive Bayes
//! 
//! `linfa-bayes` aims to provide pure Rust implementations of Naive Bayes algorithms. 
//! 
//! 
//! ## The Big Picture
//! 
//! `linfa-bayes` is a crate in the [`linfa`](https://crates.io/crates/linfa) ecosystem,
//! an effort to create a toolkit for classical Machine Learning implemented in pure Rust,
//! akin to Python's `scikit-learn`.
//! 
//! ## Current state
//! 
//! `linfa-bayes` currently provides an implementation of the following methods: 
//! 
//! - Gaussian Naive Bayes (GaussianNB)

mod error;
mod gaussian_nb;

pub use error::BayesError;
pub use gaussian_nb::GaussianNbParams;
