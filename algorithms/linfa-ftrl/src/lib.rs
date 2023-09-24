#![doc = include_str!("../README.md")]

mod algorithm;
mod error;
mod hyperparams;

use crate::hyperparams::FtrlValidParams;
pub use algorithm::Result;
pub use error::FtrlError;
pub use hyperparams::FtrlParams;
use linfa::Float;
use ndarray::Array1;
use ndarray_rand::RandomExt;
use rand::{distributions::Uniform, Rng};
use rand_xoshiro::{rand_core::SeedableRng, Xoshiro256Plus};
#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};

#[derive(Debug, Clone)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct Ftrl<F: Float> {
    /// FTRL (Follow The Regularized Leader - proximal) is a linear model for CTR prediction in online learning settings.
    /// It stores z and n values, which are later used to calculate weights at incremental model fit and during prediction.
    /// It is a special type of linear model with sigmoid function which uses L1 and L2 regularization.
    /// ```rust
    /// use linfa::Dataset;
    /// use ndarray::array;
    /// use linfa_ftrl::Ftrl;
    /// use linfa::prelude::*;
    /// let dataset = Dataset::new(array![[0.], [1.]], array![true, false]);
    /// let params = Ftrl::params();
    /// let model = params.fit_with(None, &dataset).unwrap();
    /// let predictions = model.predict(&dataset);
    /// ```
    alpha: F,
    beta: F,
    l1_ratio: F,
    l2_ratio: F,
    z: Array1<F>,
    n: Array1<F>,
}

impl<F: Float> Ftrl<F> {
    /// Create a default parameter set for construction of Follow The Regularized Leader - proximal model
    /// The description can be found [here](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41159.pdf)
    ///
    /// It requires data preprocessing done in the separate step.

    /// Create default hyperparameters. Random number generator will default to rand_xoshiro::Xoshiro256Plus
    pub fn params() -> FtrlParams<F, Xoshiro256Plus> {
        FtrlParams::default_with_rng(Xoshiro256Plus::seed_from_u64(42))
    }

    /// Create default hyperparameters with custom random number generator
    pub fn params_with_rng<R: Rng>(rng: R) -> FtrlParams<F, R> {
        FtrlParams::default_with_rng(rng)
    }

    /// Create a new model with given parameters, number of features and custom random number generator
    pub fn new<R: Rng + Clone>(params: FtrlValidParams<F, R>, nfeatures: usize) -> Ftrl<F> {
        let mut rng = params.rng.clone();
        Self {
            alpha: params.alpha,
            beta: params.beta,
            l1_ratio: params.l1_ratio,
            l2_ratio: params.l2_ratio,
            n: Array1::zeros(nfeatures),
            z: Array1::random_using(nfeatures, Uniform::new(F::zero(), F::one()), &mut rng),
        }
    }
}
