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

    pub fn fit(&self) -> FittedElasticNet {
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

#[cfg(test)]
mod tests {
    use super::soft_threshold;

    #[test]
    fn soft_threshold_works() {
        assert_eq!(soft_threshold(1.0, 1.0), 0.0);
        assert_eq!(soft_threshold(-1.0, 1.0), 0.0);
        assert_eq!(soft_threshold(0.0, 1.0), 0.0);
        assert_eq!(soft_threshold(2.0, 1.0), 1.0);
        assert_eq!(soft_threshold(-2.0, 1.0), -1.0);
    }
}
