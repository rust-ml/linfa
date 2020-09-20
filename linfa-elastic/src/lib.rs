use approx::{abs_diff_eq, abs_diff_ne};
use ndarray::{s, Array1, Array2, Axis};

/// Linear regression with both L1 and L2 regularization
///
/// Configures and minimizes the following objective function:
///             1 / (2 * n_samples) * ||y - Xw||^2_2
///             + penalty * l1_ratio * ||w||_1
///             + 0.5 * penalty * (1 - l1_ratio) * ||w||^2_2
///
pub struct ElasticNet {
    penalty: f64,
    l1_ratio: f64,
    normalization_options: NormalizationOptions,
    max_iterations: u32,
    tolerance: f64,
    random_seed: Option<u64>,
    parameter_selection: ParameterSelection,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum NormalizationOptions {
    None,
    WithIntercept,
    WithInterceptAndNormalize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ParameterSelection {
    Cyclic,
    Random,
}

pub struct FittedElasticNet {
    parameters: Array1<f64>,
    intercept: f64,
}

impl ElasticNet {
    pub fn new() -> ElasticNet {
        ElasticNet {
            penalty: 1.0,
            l1_ratio: 0.5,
            normalization_options: NormalizationOptions::WithIntercept,
            max_iterations: 1000,
            tolerance: 1e-4,
            random_seed: None,
            parameter_selection: ParameterSelection::Cyclic,
        }
    }

    pub fn penalty(mut self, penalty: f64) -> Self {
        self.penalty = penalty;
        self
    }

    pub fn l1_ratio(mut self, l1_ratio: f64) -> Self {
        self.l1_ratio = l1_ratio;
        self
    }

    pub fn with_intercept(mut self, with_intercept: bool) -> Self {
        self.normalization_options = if with_intercept {
            NormalizationOptions::WithIntercept
        } else {
            NormalizationOptions::None
        };
        self
    }

    pub fn with_intercept_and_normalize(mut self) -> Self {
        self.normalization_options = NormalizationOptions::WithInterceptAndNormalize;
        self
    }

    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    pub fn max_iterations(mut self, max_iterations: u32) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    pub fn random_seed(mut self, random_seed: u64) -> Self {
        self.random_seed = Some(random_seed);
        self
    }

    pub fn parameter_selection(mut self, parameter_selection: ParameterSelection) -> Self {
        self.parameter_selection = parameter_selection;
        self
    }

    pub fn fit(&self, x: &Array2<f64>, y: &Array1<f64>) -> FittedElasticNet {
        let y_mean = y.mean().unwrap();
        let y_centered = y - y_mean;
        // let x_scale = x.map_axis(Axis(0), |col| {
        //     let sum_sq = col.fold(0.0, |sum_sq, x| sum_sq + x * x);
        //     // If stddev for a feature is 0.0, replace it with 1.0 to make
        //     // scaling work in all cases (all entries are 0.0 anyway, we're
        //     // not changing anything, we're just avoiding to produce NaNs)
        //     if abs_diff_eq!(sum_sq, 0.0) {
        //         1.0
        //     } else {
        //         sum_sq
        //     }
        // });
        // let x_scaled = x / &x_scale;
        // eprintln!("y_centered = {}, x_scaled = {}", y_centered, x_scaled);
        // eprintln!(
        //     "x_scaled.var() = {}",
        //     x_scaled.map_axis(Axis(0), |col| col.central_moment(2).unwrap())
        // );
        let opt_result = coordinate_descent(
            &x,
            &y_centered,
            self.tolerance,
            self.max_iterations,
            self.l1_ratio,
            self.penalty,
        );
        let parameters = opt_result.0; // / &x_scale;
        let intercept = y_mean;
        FittedElasticNet {
            intercept,
            parameters,
        }
    }
}

impl Default for ElasticNet {
    fn default() -> Self {
        ElasticNet::new()
    }
}

impl FittedElasticNet {
    pub fn predict(&self) {}
    pub fn parameters(&self) -> &Array1<f64> {
        &self.parameters
    }
    pub fn intercept(&self) -> f64 {
        self.intercept
    }
}

fn coordinate_descent(
    x: &Array2<f64>,
    y: &Array1<f64>,
    tol: f64,
    max_steps: u32,
    l1_ratio: f64,
    penalty: f64,
) -> (Array1<f64>, u32) {
    let n_samples = x.shape()[0] as f64;
    let n_features = x.shape()[1];
    // the parameters of the model
    let mut w = Array1::<f64>::zeros(n_features);
    // the residuals: `y - X*w` (since w=0, this is just `y` for now),
    // the residuals are updated during the algorithm as the parameters change
    let mut r = y.clone();
    let mut n_steps = 0u32;
    let norm_cols_x = x.map_axis(Axis(0), |col| col.fold(0.0, |sum_sq, x| sum_sq + x * x));
    let mut d_w_max = f64::INFINITY;
    while n_steps < max_steps && d_w_max > tol {
        d_w_max = 0.0;
        for ii in 0..n_features {
            if abs_diff_eq!(norm_cols_x[ii], 0.0) {
                continue;
            }
            let w_ii = w[ii];
            if abs_diff_ne!(w_ii, 0.0) {
                r += &(w_ii * &x.slice(s![.., ii]));
            }
            let tmp: f64 = x.slice(s![.., ii]).dot(&r);
            w[ii] = tmp.signum() * f64::max(tmp.abs() - n_samples * l1_ratio * penalty, 0.0)
                / (norm_cols_x[ii] + n_samples * (1.0 - l1_ratio) * penalty);
            if w[ii] != 0.0 {
                r -= &(w[ii] * &x.slice(s![.., ii]));
            }
            let d_w_ii = (w[ii] - w_ii).abs();
            d_w_max = f64::max(d_w_max, d_w_ii);
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
        let model = ElasticNet::new().l1_ratio(1.0).penalty(0.1).fit(&x, &y);
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
        let model = ElasticNet::new().l1_ratio(1.0).penalty(1e-8).fit(&x, &y);
        assert_abs_diff_eq!(model.intercept(), 0.0);
        assert_abs_diff_eq!(model.parameters(), &array![1.0], epsilon = 1e-6);
        // let t = array![[2.0], [3.0], [4.0]];
        // assert_abs_diff_eq!(model.predict(&t), &t);
        // Still have to port this from python:
        // # assert_almost_equal(clf.dual_gap_, 0)
        //
        let model = ElasticNet::new().l1_ratio(1.0).penalty(0.1).fit(&x, &y);
        assert_abs_diff_eq!(model.intercept(), 0.0);
        assert_abs_diff_eq!(model.parameters(), &array![0.85], epsilon = 1e-6);
        // Still to implement
        // pred = clf.predict(T)
        // assert_array_almost_equal(pred, [1.7, 2.55, 3.4])
        // assert_almost_equal(clf.dual_gap_, 0)

        let model = ElasticNet::new().l1_ratio(1.0).penalty(0.5).fit(&x, &y);
        assert_abs_diff_eq!(model.intercept(), 0.0);
        assert_abs_diff_eq!(model.parameters(), &array![0.25], epsilon = 1e-6);
        // Still to implement
        // pred = clf.predict(T)
        // assert_array_almost_equal(pred, [0.5, 0.75, 1.])
        // assert_almost_equal(clf.dual_gap_, 0)

        let model = ElasticNet::new().l1_ratio(1.0).penalty(1.0).fit(&x, &y);
        assert_abs_diff_eq!(model.intercept(), 0.0);
        assert_abs_diff_eq!(model.parameters(), &array![0.0], epsilon = 1e-6);
        // Still to implement
        // pred = clf.predict(T)
        // assert_array_almost_equal(pred, [0, 0, 0])
        // assert_almost_equal(clf.dual_gap_, 0)
    }

    #[test]
    fn elastic_net_toy_example_works() {
        let x = array![[-1.0], [0.0], [1.0]];
        let y = array![-1.0, 0.0, 1.0];
        let model = ElasticNet::new().l1_ratio(0.3).penalty(0.5).fit(&x, &y);
        assert_abs_diff_eq!(model.intercept(), 0.0);
        assert_abs_diff_eq!(model.parameters(), &array![0.50819], epsilon = 1e-3);
        // assert_array_almost_equal(pred, [1.0163, 1.5245, 2.0327], decimal=3)
        // assert_almost_equal(clf.dual_gap_, 0)

        let model = ElasticNet::new().l1_ratio(0.5).penalty(0.5).fit(&x, &y);
        assert_abs_diff_eq!(model.intercept(), 0.0);
        assert_abs_diff_eq!(model.parameters(), &array![0.45454], epsilon = 1e-3);
        // pred = clf.predict(T)
        // assert_array_almost_equal(pred, [0.9090, 1.3636, 1.8181], 3)
        // assert_almost_equal(clf.dual_gap_, 0)
    }

    #[test]
    fn elastic_net_2d_toy_example_works() {
        let x = array![[1.0, 0.0], [0.0, 1.0]];
        let y = array![3.0, 2.0];
        let model = ElasticNet::new().penalty(0.0).fit(&x, &y);
        assert_abs_diff_eq!(model.intercept(), 2.5);
        assert_abs_diff_eq!(model.parameters(), &array![0.5, -0.5], epsilon = 0.001);
    }
}
