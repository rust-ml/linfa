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

pub mod count_vectorizer;
pub mod error;
mod helpers;
pub mod linear_scaler;
pub mod norm_scaler;

pub trait Float: linfa::Float + ndarray_linalg::Lapack + approx::AbsDiffEq {}

impl Float for f32 {}
impl Float for f64 {}
