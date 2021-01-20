use linfa::Float;
use ndarray::{ArrayView1, CowArray, Ix1};

use super::{Error, Result};

/// Linear regression with both L1 and L2 regularization
///
/// Configures and minimizes the following objective function:
///             1 / (2 * n_samples) * ||y - Xw||^2_2
///             + penalty * l1_ratio * ||w||_1
///             + 0.5 * penalty * (1 - l1_ratio) * ||w||^2_2
///
pub struct ElasticNetParams<F> {
    pub penalty: F,
    pub l1_ratio: F,
    pub with_intercept: bool,
    pub max_iterations: u32,
    pub tolerance: F,
}

///AbsDiffEq + Float + FromPrimitive + ScalarOperand + NumAssignOps>
/// Configure and fit a Elastic Net model
impl<F: Float> ElasticNetParams<F> {
    /// Create default elastic net hyper parameters
    ///
    /// By default, an intercept will be fitted. To disable fitting an
    /// intercept, call `.with_intercept(false)` before calling `.fit()`.
    ///
    /// To additionally normalize the feature matrix before fitting, call
    /// `fit_intercept_and_normalize()` before calling `fit()`. The feature
    /// matrix will not be normalized by default.
    pub fn new() -> ElasticNetParams<F> {
        ElasticNetParams {
            penalty: F::one(),
            l1_ratio: F::from(0.5).unwrap(),
            with_intercept: true,
            max_iterations: 1000,
            tolerance: F::from(1e-4).unwrap(),
        }
    }

    /// Set the overall parameter penalty parameter of the elastic net.
    /// Use `l1_ratio` to configure how the penalty distributed to L1 and L2
    /// regularization.
    pub fn penalty(mut self, penalty: F) -> Self {
        self.penalty = penalty;
        self
    }

    /// Set l1_ratio parameter of the elastic net. Controls how the parameter
    /// penalty is distributed to L1 and L2 regularization.
    /// Setting `l1_ratio` to 1.0 is equivalent to a "Lasso" penalization,
    /// setting it to 0.0 is equivalent to "Ridge" penalization.
    ///
    /// Defaults to `0.5` if not set
    ///
    /// `l1_ratio` must be between `0.0` and `1.0`.
    pub fn l1_ratio(mut self, l1_ratio: F) -> Self {
        if l1_ratio < F::zero() || l1_ratio > F::one() {
            panic!("Invalid value for l1_ratio, needs to be between 0.0 and 1.0");
        }
        self.l1_ratio = l1_ratio;
        self
    }

    /// Configure the elastic net model to fit an intercept.
    /// Defaults to `true` if not set.
    pub fn with_intercept(mut self, with_intercept: bool) -> Self {
        self.with_intercept = with_intercept;
        self
    }

    /// Set the tolerance which is the minimum absolute change in any of the
    /// model parameters needed for the parameter optimization to continue.
    ///
    /// Defaults to `1e-4` if not set
    pub fn tolerance(mut self, tolerance: F) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Set the maximum number of iterations for the optimization routine.
    ///
    /// Defaults to `1000` if not set
    pub fn max_iterations(mut self, max_iterations: u32) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Compute the intercept as the mean of `y` and center `y` if an intercept should
    /// be used, use `0.0` as intercept and leave `y` unchanged otherwise.
    pub fn compute_intercept<'a>(&self, y: ArrayView1<'a, F>) -> (F, CowArray<'a, F, Ix1>) {
        if self.with_intercept {
            let y_mean = y.mean().unwrap();
            let y_centered = &y - y_mean;
            (y_mean, y_centered.into())
        } else {
            (F::zero(), y.into())
        }
    }

    /// Validate the hyper parameters
    ///
    /// This function is called in `Self::fit` and validates all hyper parameters
    pub fn validate_params(&self) -> Result<()> {
        match self {
            ElasticNetParams { penalty, .. } if penalty.is_negative() => Err(Error::InvalidParams(
                format!("Penalty should be positive, but is {}", penalty),
            )),
            ElasticNetParams { tolerance, .. } if tolerance.is_negative() => {
                Err(Error::InvalidParams(format!(
                    "Tolerance should be positive, but is {}",
                    tolerance
                )))
            }
            ElasticNetParams { l1_ratio, .. } if l1_ratio.is_negative() || l1_ratio > &F::one() => {
                Err(Error::InvalidParams(format!(
                    "L1 ratio should be in range [0, 1], but is {}",
                    l1_ratio
                )))
            }
            _ => Ok(()),
        }
    }
}
