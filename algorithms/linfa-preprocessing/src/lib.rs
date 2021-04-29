//! # Preprocessing
//! ## The Big Picture
//!
//! `linfa-preprocessing` is a crate in the [`linfa`](https://crates.io/crates/linfa) ecosystem, an effort to create a toolkit for classical Machine Learning implemented in pure Rust, akin to Python's `scikit-learn`.
//!
//! ## Current state
//! `linfa-preprocessing` provides a pure Rust implementation of:
//! * Standard scaling
//! * Min-max scaling
//! * Max Abs Scaling
//! * Normalization (l1, l2 and max norm)
//! * Count vectorization
//! * Term frequency - inverse document frequency count vectorization
//! * Whitening

pub mod count_vectorization;
pub mod error;
mod helpers;
pub mod linear_scaling;
pub mod norm_scaling;
pub mod tf_idf_vectorization;
pub mod whitening;
