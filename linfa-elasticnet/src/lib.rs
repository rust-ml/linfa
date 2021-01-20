//! # Elastic Net
//!
//! This library contains an elastic net implementation for linear regression models. It linearily
//! combines l1 and l2 penalties of the lasso and ridge methods and offers therefore a greater
//! flexibility for feature selection. With increasing penalization certain parameters become zero,
//! their corresponding variables are dropped from the model.
//!
//! See also:
//!  * [Wikipedia on Elastic net](https://en.wikipedia.org/wiki/Elastic_net_regularization)
//!
//! ## Example
//!
//! ```
//! use linfa::traits::Fit;
//! use linfa_elasticnet::{ElasticNet, Result};
//!
//! fn main() -> Result<()> {
//!     let dataset = linfa_datasets::diabetes();
//!
//!     let model = ElasticNet::params()
//!         .l1_ratio(0.8)
//!         .penalty(0.3)
//!         .fit(&dataset)?;
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Implementation
//!
//! The coordinate descent algorithm is used to solve the lasso and ridge problem. It optimizes
//! each parameter seperately, holding all the others fixed. This cycles as long as the
//! coefficients have not stabilized or the maximum number of iterations is reached.
//!
//! See also:
//! * [Talk on Fast Regularization Paths](https://web.stanford.edu/~hastie/TALKS/glmnet.pdf)
//! * [Regularization Paths for Generalized Linear Models via Coordinate
//! Descent](http://www.jstatsoft.org/v33/i01/paper)

use linfa::Float;
use ndarray::Array1;

mod algorithm;
mod error;
mod hyperparameters;

pub use error::{Error, Result};
pub use hyperparameters::ElasticNetParams;

/// Elastic Net model
///
/// This struct contains the parameters of a fitted elastic net model. This includes the seperating
/// hyperplane, (optionally) intercept, duality gaps and the number of step needed in the
/// computation.
pub struct ElasticNet<F> {
    parameters: Array1<F>,
    intercept: F,
    duality_gap: F,
    n_steps: u32,
    variance: Result<Array1<F>>,
}

impl<F: Float> ElasticNet<F> {
    /// Create a default elastic net model
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

    /// Create a ridge model
    pub fn ridge() -> ElasticNetParams<F> {
        ElasticNetParams::new().l1_ratio(F::zero())
    }

    /// Create a lasso model
    pub fn lasso() -> ElasticNetParams<F> {
        ElasticNetParams::new().l1_ratio(F::one())
    }
}
