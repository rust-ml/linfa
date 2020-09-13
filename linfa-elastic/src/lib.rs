use ndarray::{s, Array1, Array2};

pub trait Float {}

/// Linear regression with both L1 and L2 regularization
///
/// Configures and minimizes the following objective function:
///             1 / (2 * n_samples) * ||y - Xw||^2_2
///             + alpha * l1_ratio * ||w||_1
///             + 0.5 * alpha * (1 - l1_ratio) * ||w||^2_2
/// where:
///
///             alpha = a + b and l1_ration = a / (a + b)
///
pub struct ElasticNet {
    alpha: f64,
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
            alpha: 1.0,
            l1_ratio: 0.5,
            normalization_options: NormalizationOptions::WithIntercept,
            max_iterations: 1000,
            tolerance: 1e-4,
            random_seed: None,
            parameter_selection: ParameterSelection::Cyclic,
        }
    }

    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
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
        let intercept = 0.0;
        let opt_result = coordinate_descent(
            &x,
            &y,
            self.tolerance,
            self.max_iterations,
            self.l1_ratio,
            self.alpha,
        );
        FittedElasticNet {
            intercept,
            parameters: opt_result.0,
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

fn soft_threshold(z: f64, gamma: f64) -> f64 {
    if gamma >= z.abs() {
        0.0
    } else if z > 0.0 {
        z - gamma
    } else {
        z + gamma
    }
}

/// Computes a single element-wise parameter update
/// S(\sum_i(x_ij * r_i) + beta_j, lambda * alpha)
///     / (1 + lambda * (1 - alpha))
/// where `S` is the soft_threshold function defined above
fn next_beta_j(beta_j: f64, sum_xij_ri: f64, lambda: f64, alpha_lambda: f64) -> f64 {
    // quotient could be pre-computed
    soft_threshold(sum_xij_ri + beta_j, alpha_lambda) / (1.0 + lambda - alpha_lambda)
}

/// Compute one step of parameter updates
fn step(
    x: &Array2<f64>,
    y: &Array1<f64>,
    beta: &Array1<f64>,
    alpha: f64,
    lambda: f64,
) -> (Array1<f64>, f64) {
    let alpha_lambda = alpha * lambda;
    let mut r = y - &x.dot(beta);
    let mut next_beta = beta.clone();
    let n = beta.len();
    let mut max_change = 0.0;
    for j in 0..n {
        let sum_xij_ri = x.slice(s![.., j]).dot(&r) / n as f64;
        let beta_j = next_beta_j(beta[j], sum_xij_ri, lambda, alpha_lambda);
        if beta_j != beta[j] {
            let abs_change = (beta_j - beta[j]).abs();
            if abs_change > max_change {
                max_change = abs_change;
            }
            next_beta[j] = beta_j;
            r = y - &x.dot(&next_beta);
        }
    }
    (next_beta, max_change)
}

// FIXME: only handles standardized and centered problems
fn coordinate_descent(
    x: &Array2<f64>,
    y: &Array1<f64>,
    tol: f64,
    max_steps: u32,
    alpha: f64,
    lambda: f64,
) -> (Array1<f64>, u32) {
    let mut beta = Array1::zeros(x.shape()[1]);
    let mut num_steps = 0u32;
    let mut max_change = f64::INFINITY;
    while max_change > tol && num_steps < max_steps {
        let step_result = step(x, y, &beta, alpha, lambda);
        beta = step_result.0;
        max_change = step_result.1;
        num_steps += 1;
    }
    (beta, num_steps)
}

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

#[cfg(test)]
mod tests {
    use super::{
        coordinate_descent, elastic_net_objective, elastic_net_penalty, soft_threshold,
        squared_error, step, ElasticNet,
    };
    use approx::{abs_diff_eq, assert_abs_diff_eq};
    use ndarray::array;

    #[test]
    fn soft_threshold_works() {
        abs_diff_eq!(soft_threshold(1.0, 1.0), 0.0);
        abs_diff_eq!(soft_threshold(-1.0, 1.0), 0.0);
        abs_diff_eq!(soft_threshold(0.0, 1.0), 0.0);
        abs_diff_eq!(soft_threshold(2.0, 1.0), 1.0);
        abs_diff_eq!(soft_threshold(-2.0, 1.0), -1.0);
    }

    #[test]
    fn step_does_something() {
        let x = array![[1., 2.], [3., 4.]];
        let y = array![1., 2.];
        let beta = array![0., 0.];
        let next_beta = step(&x, &y, &beta, 0.5, 1.0).0;
        assert_ne!(beta, next_beta);
    }

    #[test]
    fn elastic_net_penalty_works() {
        let beta = array![-2.0, 1.0];
        abs_diff_eq!(elastic_net_penalty(&beta, 0.8), 0.4 + 0.1 + 1.6 + 0.8);
        abs_diff_eq!(elastic_net_penalty(&beta, 1.0), 3.0);
        abs_diff_eq!(elastic_net_penalty(&beta, 0.0), 2.5);

        let beta2 = array![0.0, 0.0];
        abs_diff_eq!(elastic_net_penalty(&beta2, 0.8), 0.0);
        abs_diff_eq!(elastic_net_penalty(&beta2, 1.0), 0.0);
        abs_diff_eq!(elastic_net_penalty(&beta2, 0.0), 0.0);
    }

    #[test]
    fn squared_error_works() {
        let x = array![[2.0, 1.0], [-1.0, 2.0]];
        let y = array![1.0, 1.0];
        let beta = array![0.0, 1.0];
        abs_diff_eq!(squared_error(&x, &y, 0.0, &beta), 0.25);
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
        println!(
            "num_steps = {}\nnew_beta = {:?}",
            opt_result.1, opt_result.0
        );
        let objective_end = elastic_net_objective(&x, &y, intercept, &opt_result.0, alpha, lambda);
        assert!(objective_start > objective_end);
    }

    #[test]
    fn simple_elastic_net_works() {
        let x = array![[1.0, 0.0], [0.0, 1.0]];
        let y = array![1.0, -1.0];
        let model = ElasticNet::new().alpha(0.0001).l1_ratio(0.8).fit(&x, &y);
        assert_abs_diff_eq!(model.intercept(), 0.0);
        assert_abs_diff_eq!(model.parameters(), &array![1.0, -1.0], epsilon = 0.001);
    }
}
