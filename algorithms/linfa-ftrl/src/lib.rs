#![doc = include_str!("../README.md")]

mod algorithm;
mod error;
mod hyperparams;

use crate::hyperparams::{FtrlParams, FtrlValidParams};
pub use algorithm::Result;
pub use error::FtrlError;
use linfa::Float;
use ndarray::Array1;
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;

#[derive(Clone, Debug)]
pub struct FTRL<F: Float> {
    /// FTRL (Follow The Regularized Leader - proximal) is a linear model for CTR prediction in online learning settings.
    /// It stores z and n values, which are later used to calculate weights at incremental model fit and during prediction.
    /// It is a special type of linear model with sigmoid function which uses L1 and L2 regularization.
    /// ```rust
    /// use linfa::Dataset;
    /// use ndarray::array;
    /// use linfa_ftrl::FTRL;
    /// use linfa::prelude::*;
    /// let dataset = Dataset::new(array![[0.], [1.]], array![true, false]);
    /// let params = FTRL::params();
    /// let model = params.fit_with(None, &dataset).unwrap();
    /// let predictions = model.predict(&dataset);
    /// ```
    params: FtrlValidParams<F>,
    z: Array1<F>,
    n: Array1<F>,
}

impl<F: Float> FTRL<F> {
    /// Create a default parameter set for construction of Follow The Regularized Leader - proximal model
    /// The description can be found here https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41159.pdf
    ///
    /// It requires data preprocessing done in the separate step.

    /// Create default hyperparameters
    pub fn params() -> FtrlParams<F> {
        FtrlParams::default()
    }

    /// Create a new model with given parameters and number of features
    pub fn new(params: FtrlValidParams<F>, nfeatures: usize) -> FTRL<F> {
        Self {
            params,
            n: Array1::zeros(nfeatures),
            z: Array1::random(nfeatures, Uniform::new(F::zero(), F::one())),
        }
    }
}
