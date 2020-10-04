use std::borrow::Cow;

use approx::{abs_diff_eq, abs_diff_ne, AbsDiffEq};
use ndarray::{s, Array1, Array2, ArrayView1, Axis, ScalarOperand};
use num_traits::{Float, FromPrimitive, NumAssignOps};

/// Linear regression with both L1 and L2 regularization
///
/// Configures and minimizes the following objective function:
///             1 / (2 * n_samples) * ||y - Xw||^2_2
///             + penalty * l1_ratio * ||w||_1
///             + 0.5 * penalty * (1 - l1_ratio) * ||w||^2_2
///
pub struct ElasticNet<F> {
    penalty: F,
    l1_ratio: F,
    with_intercept: bool,
    max_iterations: u32,
    tolerance: F,
}

/// A fitted elastic net which can be used for making predictions
pub struct FittedElasticNet<F> {
    parameters: Array1<F>,
    intercept: F,
}

/// Configure and fit a Elastic Net model
impl<F: AbsDiffEq + Float + FromPrimitive + ScalarOperand + NumAssignOps> ElasticNet<F> {
    /// Create a default elastic net model
    ///
    /// By default, an intercept will be fitted. To disable fitting an
    /// intercept, call `.with_intercept(false)` before calling `.fit()`.
    ///
    /// To additionally normalize the feature matrix before fitting, call
    /// `fit_intercept_and_normalize()` before calling `fit()`. The feature
    /// matrix will not be normalized by default.
    pub fn new() -> ElasticNet<F> {
        ElasticNet {
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

    /// Fit an elastic net model given a feature matrix `x` and a target
    /// variable `y`.
    ///
    /// The feature matrix `x` must have shape `(n_samples, n_features)`
    ///
    /// The target variable `y` must have shape `(n_samples)`
    ///
    /// Returns a `FittedElasticNet` object which contains the fitted
    /// parameters and can be used to `predict` values of the target variable
    /// for new feature values.
    pub fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<FittedElasticNet<F>, String> {
        let (intercept, y) = self.compute_intercept(y);
        let (parameters, _) = coordinate_descent(
            &x,
            &y,
            self.tolerance,
            self.max_iterations,
            self.l1_ratio,
            self.penalty,
        );
        Ok(FittedElasticNet {
            intercept,
            parameters,
        })
    }

    /// Compute the intercept as the mean of `y` and center `y` if an intercept should
    /// be used, use `0.0` as intercept and leave `y` unchanged otherwise.
    fn compute_intercept<'a>(&self, y: &'a Array1<F>) -> (F, Cow<'a, Array1<F>>) {
        if self.with_intercept {
            let y_mean = y.mean().unwrap();
            let y_centered = y - y_mean;
            (y_mean, Cow::Owned(y_centered))
        } else {
            (F::zero(), Cow::Borrowed(y))
        }
    }
}

impl<F: AbsDiffEq + Float + FromPrimitive + ScalarOperand + NumAssignOps> Default
    for ElasticNet<F>
{
    fn default() -> Self {
        ElasticNet::new()
    }
}

/// View the fitted parameters and make predictions with a fitted
/// elastic net model
impl<F: Float + FromPrimitive + ScalarOperand> FittedElasticNet<F> {
    /// Given an input matrix `X`, with shape `(n_samples, n_features)`,
    /// `predict` returns the target variable according to elastic net
    /// learned from the training data distribution.
    pub fn predict(&self, x: &Array2<F>) -> Array1<F> {
        x.dot(&self.parameters) + self.intercept
    }

    /// Get the fitted parameters
    pub fn parameters(&self) -> &Array1<F> {
        &self.parameters
    }

    /// Get the fitted intercept, 0. if no intercept was fitted
    pub fn intercept(&self) -> F {
        self.intercept
    }
}

fn coordinate_descent<F: AbsDiffEq + Float + FromPrimitive + ScalarOperand + NumAssignOps>(
    x: &Array2<F>,
    y: &Array1<F>,
    tol: F,
    max_steps: u32,
    l1_ratio: F,
    penalty: F,
) -> (Array1<F>, u32) {
    let n_samples = F::from(x.shape()[0]).unwrap();
    let n_features = x.shape()[1];
    // the parameters of the model
    let mut w = Array1::<F>::zeros(n_features);
    // the residuals: `y - X*w` (since w=0, this is just `y` for now),
    // the residuals are updated during the algorithm as the parameters change
    let mut r = y.clone();
    let mut n_steps = 0u32;
    let norm_cols_x = x.map_axis(Axis(0), |col| {
        col.fold(F::zero(), |sum_sq, &x| sum_sq + x * x)
    });
    let mut d_w_max = F::infinity();
    while n_steps < max_steps && d_w_max > tol {
        d_w_max = F::zero();
        for ii in 0..n_features {
            if abs_diff_eq!(norm_cols_x[ii], F::zero()) {
                continue;
            }
            let w_ii = w[ii];
            if abs_diff_ne!(w_ii, F::zero()) {
                let slc: ArrayView1<F> = x.slice(s![.., ii]);
                r += &(&slc * w_ii);
            }
            let tmp: F = x.slice(s![.., ii]).dot(&r);
            w[ii] = tmp.signum() * F::max(tmp.abs() - n_samples * l1_ratio * penalty, F::zero())
                / (norm_cols_x[ii] + n_samples * (F::one() - l1_ratio) * penalty);
            if w[ii] != F::zero() {
                let slc: ArrayView1<F> = x.slice(s![.., ii]);
                r -= &(&slc * w[ii]);
            }
            let d_w_ii = (w[ii] - w_ii).abs();
            d_w_max = F::max(d_w_max, d_w_ii);
        }
        n_steps += 1;
    }
    (w, n_steps)
}

#[cfg(test)]
mod tests {
    use super::{coordinate_descent, ElasticNet};
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Array1, Array2};

    fn elastic_net_objective(
        x: &Array2<f64>,
        y: &Array1<f64>,
        intercept: f64,
        beta: &Array1<f64>,
        alpha: f64,
        lambda: f64,
    ) -> f64 {
        squared_error(x, y, intercept, beta) + lambda * elastic_net_penalty(beta, alpha)
    }

    fn squared_error(x: &Array2<f64>, y: &Array1<f64>, intercept: f64, beta: &Array1<f64>) -> f64 {
        let mut resid = -x.dot(beta);
        resid -= intercept;
        resid += y;
        let mut result = 0.0;
        for r in &resid {
            result += r * r;
        }
        result /= 2.0 * y.len() as f64;
        result
    }

    fn elastic_net_penalty(beta: &Array1<f64>, alpha: f64) -> f64 {
        let mut penalty = 0.0;
        for beta_j in beta {
            penalty += (1.0 - alpha) / 2.0 * beta_j * beta_j + alpha * beta_j.abs();
        }
        penalty
    }

    #[test]
    fn elastic_net_penalty_works() {
        let beta = array![-2.0, 1.0];
        assert_abs_diff_eq!(
            elastic_net_penalty(&beta, 0.8),
            0.4 + 0.1 + 1.6 + 0.8,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(elastic_net_penalty(&beta, 1.0), 3.0);
        assert_abs_diff_eq!(elastic_net_penalty(&beta, 0.0), 2.5);

        let beta2 = array![0.0, 0.0];
        assert_abs_diff_eq!(elastic_net_penalty(&beta2, 0.8), 0.0);
        assert_abs_diff_eq!(elastic_net_penalty(&beta2, 1.0), 0.0);
        assert_abs_diff_eq!(elastic_net_penalty(&beta2, 0.0), 0.0);
    }

    #[test]
    fn squared_error_works() {
        let x = array![[2.0, 1.0], [-1.0, 2.0]];
        let y = array![1.0, 1.0];
        let beta = array![0.0, 1.0];
        assert_abs_diff_eq!(squared_error(&x, &y, 0.0, &beta), 0.25);
    }

    #[test]
    fn coordinate_descent_lowers_objective() {
        let x = array![[1.0, 0.0], [0.0, 1.0]];
        let y = array![1.0, -1.0];
        let beta = array![0.0, 0.0];
        let intercept = 0.0;
        let alpha = 0.8;
        let lambda = 0.001;
        let objective_start = elastic_net_objective(&x, &y, intercept, &beta, alpha, lambda);
        let opt_result = coordinate_descent(&x, &y, 1e-4, 3, alpha, lambda);
        let objective_end = elastic_net_objective(&x, &y, intercept, &opt_result.0, alpha, lambda);
        assert!(objective_start > objective_end);
    }

    #[test]
    fn lasso_zero_works() {
        let x = array![[0.], [0.], [0.]];
        let y = array![0., 0., 0.];
        let model = ElasticNet::new()
            .l1_ratio(1.0)
            .penalty(0.1)
            .fit(&x, &y)
            .unwrap();
        assert_abs_diff_eq!(model.intercept(), 0.);
        assert_abs_diff_eq!(model.parameters(), &array![0.]);
    }

    #[test]
    fn lasso_toy_example_works() {
        // Test Lasso on a toy example for various values of alpha.
        // When validating this against glmnet notice that glmnet divides it
        // against n_samples.
        let x = array![[-1.0], [0.0], [1.0]];
        let y = array![-1.0, 0.0, 1.0];
        // input for prediction
        let t = array![[2.0], [3.0], [4.0]];
        let model = ElasticNet::new()
            .l1_ratio(1.0)
            .penalty(1e-8)
            .fit(&x, &y)
            .unwrap();
        assert_abs_diff_eq!(model.intercept(), 0.0);
        assert_abs_diff_eq!(model.parameters(), &array![1.0], epsilon = 1e-6);
        assert_abs_diff_eq!(model.predict(&t), array![2.0, 3.0, 4.0], epsilon = 1e-6);
        // Still have to port this from python:
        // # assert_almost_equal(clf.dual_gap_, 0)

        let model = ElasticNet::new()
            .l1_ratio(1.0)
            .penalty(0.1)
            .fit(&x, &y)
            .unwrap();
        assert_abs_diff_eq!(model.intercept(), 0.0);
        assert_abs_diff_eq!(model.parameters(), &array![0.85], epsilon = 1e-6);
        assert_abs_diff_eq!(model.predict(&t), array![1.7, 2.55, 3.4], epsilon = 1e-6);
        // Still to implement
        // assert_almost_equal(clf.dual_gap_, 0)

        let model = ElasticNet::new()
            .l1_ratio(1.0)
            .penalty(0.5)
            .fit(&x, &y)
            .unwrap();
        assert_abs_diff_eq!(model.intercept(), 0.0);
        assert_abs_diff_eq!(model.parameters(), &array![0.25], epsilon = 1e-6);
        assert_abs_diff_eq!(model.predict(&t), array![0.5, 0.75, 1.0], epsilon = 1e-6);
        // Still to implement
        // assert_almost_equal(clf.dual_gap_, 0)

        let model = ElasticNet::new()
            .l1_ratio(1.0)
            .penalty(1.0)
            .fit(&x, &y)
            .unwrap();
        assert_abs_diff_eq!(model.intercept(), 0.0);
        assert_abs_diff_eq!(model.parameters(), &array![0.0], epsilon = 1e-6);
        assert_abs_diff_eq!(model.predict(&t), array![0.0, 0.0, 0.0], epsilon = 1e-6);
        // Still to implement
        // assert_almost_equal(clf.dual_gap_, 0)
    }

    #[test]
    fn elastic_net_toy_example_works() {
        let x = array![[-1.0], [0.0], [1.0]];
        let y = array![-1.0, 0.0, 1.0];
        // for predictions
        let t = array![[2.0], [3.0], [4.0]];
        let model = ElasticNet::new()
            .l1_ratio(0.3)
            .penalty(0.5)
            .fit(&x, &y)
            .unwrap();
        assert_abs_diff_eq!(model.intercept(), 0.0);
        assert_abs_diff_eq!(model.parameters(), &array![0.50819], epsilon = 1e-3);
        assert_abs_diff_eq!(
            model.predict(&t),
            array![1.0163, 1.5245, 2.0327],
            epsilon = 1e-3
        );
        // assert_almost_equal(clf.dual_gap_, 0)

        let model = ElasticNet::new()
            .l1_ratio(0.5)
            .penalty(0.5)
            .fit(&x, &y)
            .unwrap();
        assert_abs_diff_eq!(model.intercept(), 0.0);
        assert_abs_diff_eq!(model.parameters(), &array![0.45454], epsilon = 1e-3);
        assert_abs_diff_eq!(
            model.predict(&t),
            array![0.9090, 1.3636, 1.8181],
            epsilon = 1e-3
        );
        // assert_almost_equal(clf.dual_gap_, 0)
    }

    #[test]
    fn elastic_net_2d_toy_example_works() {
        let x = array![[1.0, 0.0], [0.0, 1.0]];
        let y = array![3.0, 2.0];
        let model = ElasticNet::new().penalty(0.0).fit(&x, &y).unwrap();
        assert_abs_diff_eq!(model.intercept(), 2.5);
        assert_abs_diff_eq!(model.parameters(), &array![0.5, -0.5], epsilon = 0.001);
    }
}
