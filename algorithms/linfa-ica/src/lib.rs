//! # Independent Component Analysis (ICA)
//!
//! `linfa-ica` aims to provide pure Rust implementations of ICA algorithms.
//!
//! ICA separates mutivariate signals into their additive, independent subcomponents.
//! ICA is primarily used for separating superimposed signals and not for dimensionality
//! reduction.
//!
//! Input data is whitened (remove underlying correlation) before modeling.
//!
//! ## The Big Picture
//!
//! `linfa-bayes` is a crate in the [`linfa`](https://crates.io/crates/linfa) ecosystem,
//! an effort to create a toolkit for classical Machine Learning implemented in pure Rust,
//! akin to Python's `scikit-learn`.
//!
//! ## Current state
//!
//! `linfa-ica` currently provides an implementation of the following methods:
//!
//! - Fast Independent Component Analysis (Fast ICA)

#[macro_use]
extern crate ndarray;

pub mod error;
pub mod fast_ica;
