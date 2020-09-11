use ndarray::{Array1, Array2, s};

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
    max_iterations: u64,
    tolerance: f64,
    random_seed: Option<u64>,
    parameter_selection: ParameterSelection
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

pub struct FittedElasticNet {}

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

    pub fn max_iterations(mut self, max_iterations: u64) -> Self {
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
        FittedElasticNet {}
    }
}

impl Default for ElasticNet {
    fn default() -> Self {
        ElasticNet::new()
    }
}

impl FittedElasticNet {
    pub fn predict(&self) {}
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
fn step(x: &Array2<f64>, y: &Array1<f64>, beta: &Array1<f64>, alpha: f64, lambda: f64) -> Array1<f64> {
    let alpha_lambda = alpha * lambda;
    let mut r = y - &x.dot(beta);
    let mut next_beta = beta.clone();
    let n = beta.len();
    for j in 0..n {
        let sum_xij_ri = x.slice(s![.., j]).dot(&r) / n as f64;
        let beta_j = next_beta_j(beta[j], sum_xij_ri, lambda, alpha_lambda);
        if beta_j != beta[j] {
            next_beta[j] = beta_j;
            r = y - &x.dot(&next_beta);
        }
    }
    next_beta
}

#[cfg(test)]
mod tests {
    use super::{soft_threshold, step};
    use ndarray::array;

    #[test]
    fn soft_threshold_works() {
        assert_eq!(soft_threshold(1.0, 1.0), 0.0);
        assert_eq!(soft_threshold(-1.0, 1.0), 0.0);
        assert_eq!(soft_threshold(0.0, 1.0), 0.0);
        assert_eq!(soft_threshold(2.0, 1.0), 1.0);
        assert_eq!(soft_threshold(-2.0, 1.0), -1.0);
    }

    #[test]
    fn step_does_something() {
        let x = array![[1., 2.], [3., 4.]];
        let y = array![1., 2.];
        let beta = array![0., 0.];
        let next_beta = step(&x, &y, &beta, 0.5, 1.0);
        assert_ne!(beta, next_beta);
    }
}
