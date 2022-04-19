//! # Follow the regularized leader - proximal
//!
//! ## The Big Picture
//!
//! `linfa-ftrl` is a crate in the [`linfa`](https://crates.io/crates/linfa) ecosystem, an effort to create a toolkit for classical Machine Learning implemented in pure Rust, akin to Python's `scikit-learn`.
//!
//! ## Current state
//! `linfa-ftrl` provides a pure Rust implementation of an [algorithm](struct.FollowTheRegularizedLeader.html).
//!
//! ## Examples
//!
//! There is an usage example in the `examples/` directory. To run, use:
//!
//! ```bash
//! $ cargo run --example winequality
//! ```
//!
mod algorithm;
mod error;
mod hyperparams;

use crate::hyperparams::FtrlParams;
pub use algorithm::Result;
pub use error::FtrlError;
use linfa::Float;
use ndarray::Array1;
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;

#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};

#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct FollowTheRegularizedLeader<F: Float> {
    params: FtrlParams<F>,
    z: Array1<F>,
    n: Array1<F>,
}

impl<F: Float> FollowTheRegularizedLeader<F> {
    /// Create a default parameter set for construction of Follow The Regularized Leader - proximal model
    /// The description can be found here https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41159.pdf
    ///
    /// It requires data preprocessing done in the separate step.

    /// Create default hyperparameters
    pub fn params() -> FtrlParams<F> {
        FtrlParams::default()
    }

    /// Create a new model with given parameters and number of features
    pub fn new(params: &FtrlParams<F>, nfeatures: usize) -> FollowTheRegularizedLeader<F> {
        Self {
            params: params.clone(),
            n: Array1::zeros(nfeatures),
            z: Array1::random(nfeatures, Uniform::new(F::zero(), F::one())),
        }
    }
}
