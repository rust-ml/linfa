#![doc = include_str!("../README.md")]

use linfa::Float;
use ndarray::Array1;

#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};

mod algorithm;
mod error;
mod hyperparams;

pub use error::{ElasticNetError, Result};
pub use hyperparams::{ElasticNetParams, ElasticNetValidParams};

#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
/// Elastic Net model
///
/// This struct contains the parameters of a fitted elastic net model. This includes the seperating
/// hyperplane, (optionally) intercept, duality gaps and the number of step needed in the
/// computation.
///
/// ## Model implementation
///
/// The coordinate descent algorithm is used to solve the lasso and ridge problem. It optimizes
/// each parameter seperately, holding all the others fixed. This cycles as long as the
/// coefficients have not stabilized or the maximum number of iterations is reached.
///
/// See also:
/// * [Talk on Fast Regularization Paths](https://web.stanford.edu/~hastie/TALKS/glmnet.pdf)
/// * [Regularization Paths for Generalized Linear Models via Coordinate
/// Descent](http://www.jstatsoft.org/v33/i01/paper)
pub struct ElasticNet<F> {
    hyperplane: Array1<F>,
    intercept: F,
    duality_gap: F,
    n_steps: u32,
    variance: Result<Array1<F>>,
}

impl<F: Float> ElasticNet<F> {
    /// Create a default parameter set for construction of ElasticNet model
    ///
    /// By default, an intercept will be fitted. To disable fitting an
    /// intercept, call `.with_intercept(false)` before calling `.fit()`.
    ///
    /// To additionally normalize the feature matrix before fitting, call
    /// `fit_intercept_and_normalize()` before calling `fit()`. The feature
    /// matrix will not be normalized by default.
    pub fn params() -> ElasticNetParams<F> {
        ElasticNetParams::new()
    }

    /// Create a ridge only model
    pub fn ridge() -> ElasticNetParams<F> {
        ElasticNetParams::new().l1_ratio(F::zero())
    }

    /// Create a LASSO only model
    pub fn lasso() -> ElasticNetParams<F> {
        ElasticNetParams::new().l1_ratio(F::one())
    }
}
