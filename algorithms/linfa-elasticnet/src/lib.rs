#![doc = include_str!("../README.md")]

use linfa::Float;
use ndarray::{Array1, Array2};

#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};

mod algorithm;
mod error;
mod hyperparams;

pub use error::{ElasticNetError, Result};
pub use hyperparams::{
    ElasticNetParams, ElasticNetValidParams, MultiTaskElasticNetParams,
    MultiTaskElasticNetValidParams,
};

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

/// MultiTask Elastic Net model
///
/// This struct contains the parameters of a fitted multi-task elastic net model. This includes the
/// coefficients (a 2-dimensional array), (optionally) intercept (a 1-dimensional array), duality gaps
/// and the number of steps needed in the computation.
///
/// ## Model implementation
///
/// The block coordinate descent is widely used to solve generalized linear models optimization problems,
/// like Group Lasso, MultiTask Ridge or MultiTask Lasso. It cycles through a group of parameters and update
/// the groups separately, holding all the others fixed. The optimization routine stops when a criterion is
/// satisfied (dual sub-optimality gap or change in coefficients).
pub struct MultiTaskElasticNet<F> {
    hyperplane: Array2<F>,
    intercept: Array1<F>,
    duality_gap: F,
    n_steps: u32,
    variance: Result<Array1<F>>,
}

impl<F: Float> MultiTaskElasticNet<F> {
    pub fn params() -> MultiTaskElasticNetParams<F> {
        MultiTaskElasticNetParams::new()
    }

    /// Create a multi-task ridge only model
    pub fn ridge() -> MultiTaskElasticNetParams<F> {
        MultiTaskElasticNetParams::new().l1_ratio(F::zero())
    }

    /// Create a multi-task Lasso only model
    pub fn lasso() -> MultiTaskElasticNetParams<F> {
        MultiTaskElasticNetParams::new().l1_ratio(F::one())
    }
}
