use crate::{error::Error, Float, LogisticRegression};
use linfa::ParamGuard;
use ndarray::Array1;
use std::marker::PhantomData;

/// A two-class logistic regression model.
///
/// Logistic regression combines linear models with
/// the sigmoid function `sigm(x) = 0/(1+exp(-x))`
/// to learn a family of functions that map the feature space to `[-1,1]`.
///
/// Logistic regression is used in binary classification
/// by interpreting the predicted value as the probability that the sample
/// has label `0`. A threshold can be set in the [fitted model](struct.FittedLogisticRegression.html) to decide the minimum
/// probability needed to classify a sample as `0`, which defaults to `0.5`.
///
/// In this implementation any binary set of labels can be used, not necessarily `-1` and `1`.
///
/// l1 regularization is used by this algorithm and is weighted by parameter `alpha`. Setting `alpha`
/// close to zero removes regularization and the problem solved minimizes only the
/// empirical risk. On the other hand, setting `alpha` to a high value increases
/// the weight of the l1 norm of the linear model coefficients in the cost function.
///
/// ## Examples
///
/// Here's an example on how to train a logistic regression model on the `winequality` dataset
/// ```rust
/// use linfa::traits::{Fit, Predict};
/// use linfa_logistic::LogisticRegression;
///
/// // Example on using binary labels different from -1 and 1
/// let dataset = linfa_datasets::winequality().map_targets(|x| if *x > 5 { "good" } else { "bad" });
/// let model = LogisticRegression::params().fit(&dataset).unwrap();
/// let prediction = model.predict(&dataset);
/// ```
pub struct LogisticRegressionValidParams<F: Float, C> {
    alpha: F,
    fit_intercept: bool,
    max_iterations: u64,
    gradient_tolerance: F,
    initial_params: Option<(Array1<F>, F)>,
    phantom: PhantomData<C>,
}

impl<F: Float, C> LogisticRegressionValidParams<F, C> {
    pub fn alpha(&self) -> F {
        self.alpha
    }

    pub fn fit_intercept(&self) -> bool {
        self.fit_intercept
    }

    pub fn max_iterations(&self) -> u64 {
        self.max_iterations
    }

    pub fn gradient_tolerance(&self) -> F {
        self.gradient_tolerance
    }

    pub fn initial_params(&self) -> &Option<(Array1<F>, F)> {
        &self.initial_params
    }
}

pub struct LogisticRegressionParams<F: Float, C>(LogisticRegressionValidParams<F, C>);

impl<F: Float, C> Default for LogisticRegressionParams<F, C> {
    fn default() -> LogisticRegressionParams<F, C> {
        LogisticRegressionParams::new()
    }
}

impl<F: Float, C: PartialOrd + Clone> LogisticRegression<F, C> {
    pub fn params() -> LogisticRegressionParams<F, C> {
        LogisticRegressionParams::new()
    }
}

impl<F: Float, C> LogisticRegressionParams<F, C> {
    /// Creates a new LogisticRegression with default configuration.
    pub fn new() -> LogisticRegressionParams<F, C> {
        Self(LogisticRegressionValidParams {
            alpha: F::cast(1.0),
            fit_intercept: true,
            max_iterations: 99,
            gradient_tolerance: F::cast(1e-4),
            initial_params: None,
            phantom: PhantomData,
        })
    }

    /// Set the normalization parameter `alpha` used for L1 normalization,
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
    /// defaults to `99`.
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

    /// Configure the initial parameters from where the optimization starts.
    /// The `params` array must have the same size as the number of columns of
    /// the feature matrix `x` passed to the `fit` method
    pub fn initial_params(mut self, params: Array1<F>, intercept: F) -> Self {
        self.0.initial_params = Some((params, intercept));
        self
    }
}

impl<F: Float, C> ParamGuard for LogisticRegressionParams<F, C> {
    type Checked = LogisticRegressionValidParams<F, C>;
    type Error = Error;

    fn check_ref(&self) -> Result<&Self::Checked, Self::Error> {
        Ok(&self.0)
    }

    fn check(self) -> Result<Self::Checked, Self::Error> {
        self.check_ref()?;
        Ok(self.0)
    }
}
