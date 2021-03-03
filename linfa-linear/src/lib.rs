//!
//! `linfa-linear` aims to provide pure Rust implementations of popular linear regression algorithms.
//!
//! ## The Big Picture
//!
//! `linfa-linear` is a crate in the [`linfa`](https://crates.io/crates/linfa) ecosystem, an effort to create a toolkit for classical Machine Learning
//! implemented in pure Rust, akin to Python's `scikit-learn`.
//!
//! ## Current state
//!
//! `linfa-linear` currently provides an implementation of the following regression algorithms:
//! - Ordinary Least Squares
//! - Generalized Linear Models (GLM)
//!
//! ## Examples
//!
//! There is an usage example in the `examples/` directory. To run, use:
//!
//! ```bash
//! $ cargo run --features openblas --example diabetes
//! $ cargo run --example glm
//! ```

pub mod error;
mod float;
pub mod glm;
pub mod ols;

pub use error::Result;
pub use glm::TweedieRegressor;
pub use ols::LinearRegression;
