use hyperparameters::ElasticNetParams;
use linfa::Float;
use ndarray::Array1;

mod algorithm;
mod hyperparameters;

/// A fitted elastic net which can be used for making predictions
pub struct ElasticNet<F> {
    parameters: Array1<F>,
    intercept: F,
    duality_gap: F,
    n_steps: u32,
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
