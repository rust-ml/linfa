use linfa::ParamGuard;
use ndarray::{Array, Dimension};

use crate::error::Error;
use crate::float::Float;

/// A generalized logistic regression type that specializes as either binomial logistic regression
/// or multinomial logistic regression.
#[derive(Debug, Clone, PartialEq)]
pub struct LogisticRegressionParams<F: Float, D: Dimension>(LogisticRegressionValidParams<F, D>);

#[derive(Debug, Clone, PartialEq)]
pub struct LogisticRegressionValidParams<F: Float, D: Dimension> {
    pub(crate) alpha: F,
    pub(crate) fit_intercept: bool,
    pub(crate) max_iterations: u64,
    pub(crate) gradient_tolerance: F,
    pub(crate) initial_params: Option<Array<F, D>>,
}

impl<F: Float, D: Dimension> ParamGuard for LogisticRegressionParams<F, D> {
    type Checked = LogisticRegressionValidParams<F, D>;
    type Error = Error;

    fn check_ref(&self) -> Result<&Self::Checked, Self::Error> {
        if !self.0.alpha.is_finite() || self.0.alpha < <F as num_traits::Zero>::zero() {
            return Err(Error::InvalidAlpha);
        }
        if !self.0.gradient_tolerance.is_finite()
            || self.0.gradient_tolerance <= <F as num_traits::Zero>::zero()
        {
            return Err(Error::InvalidGradientTolerance);
        }
        if let Some(params) = self.0.initial_params.as_ref() {
            if params.iter().any(|p| !p.is_finite()) {
                return Err(Error::InvalidInitialParameters);
            }
        }
        Ok(&self.0)
    }

    fn check(self) -> Result<Self::Checked, Self::Error> {
        self.check_ref()?;
        Ok(self.0)
    }
}

impl<F: Float, D: Dimension> LogisticRegressionParams<F, D> {
    /// Creates a new LogisticRegression with default configuration.
    pub fn new() -> Self {
        Self(LogisticRegressionValidParams {
            alpha: F::cast(1.0),
            fit_intercept: true,
            max_iterations: 100,
            gradient_tolerance: F::cast(1e-4),
            initial_params: None,
        })
    }

    /// Set the regularization parameter `alpha` used for L2 regularization,
    /// defaults to `1.0`.
    pub fn alpha(mut self, alpha: F) -> Self {
        self.0.alpha = alpha;
        self
    }

    /// Configure if an intercept should be fitted, defaults to `true`.
    pub fn with_intercept(mut self, fit_intercept: bool) -> Self {
        self.0.fit_intercept = fit_intercept;
        self
    }

    /// Configure the maximum number of iterations that the solver should perform,
    /// defaults to `100`.
    pub fn max_iterations(mut self, max_iterations: u64) -> Self {
        self.0.max_iterations = max_iterations;
        self
    }

    /// Configure the minimum change to the gradient to continue the solver,
    /// defaults to `1e-4`.
    pub fn gradient_tolerance(mut self, gradient_tolerance: F) -> Self {
        self.0.gradient_tolerance = gradient_tolerance;
        self
    }

    /// Configure the initial parameters from where the optimization starts.  The `params` array
    /// must have the same number of rows as there are columns on the feature matrix `x` passed to
    /// the `fit` method. If `with_intercept` is set, then it needs to have one more row. For
    /// multinomial regression, `params` also must have the same number of columns as the number of
    /// distinct classes in `y`.
    pub fn initial_params(mut self, params: Array<F, D>) -> Self {
        self.0.initial_params = Some(params);
        self
    }
}
