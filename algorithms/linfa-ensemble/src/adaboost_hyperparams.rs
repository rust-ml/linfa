use linfa::{
    error::{Error, Result},
    ParamGuard,
};
use rand::rngs::ThreadRng;
use rand::Rng;

/// The set of valid hyperparameters for the [AdaBoost](crate::AdaBoost) algorithm.
///
/// ## Parameters
///
/// * `n_estimators`: The maximum number of weak learners to train sequentially.
///   More estimators generally improve performance but increase training time and risk overfitting.
///   Typical values range from 50 to 500. Default: 50.
///
/// * `learning_rate`: Shrinks the contribution of each classifier. There is a trade-off between
///   `learning_rate` and `n_estimators`. Lower values require more estimators to achieve the same
///   performance but may generalize better. Must be positive. Default: 1.0.
///
/// * `model_params`: The parameters for the base learner (weak classifier). Typically, shallow
///   decision trees (stumps with max_depth=1 or max_depth=2) are used as weak learners.
///
/// * `rng`: Random number generator used for tie-breaking and reproducibility.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdaBoostValidParams<P, R> {
    /// The maximum number of estimators to train
    pub n_estimators: usize,
    /// The learning rate (shrinkage parameter)
    pub learning_rate: f64,
    /// The base learner parameters
    pub model_params: P,
    /// Random number generator
    pub rng: R,
}

/// A helper struct for building [AdaBoost](crate::AdaBoost) hyperparameters.
///
/// This struct follows the builder pattern, allowing you to chain method calls to configure
/// the AdaBoost algorithm before fitting.
///
/// ## Example
///
/// ```no_run
/// use linfa_ensemble::AdaBoostParams;
/// use linfa_trees::DecisionTree;
///
/// let params = AdaBoostParams::new(DecisionTree::<f64, usize>::params().max_depth(Some(1)))
///     .n_estimators(100)
///     .learning_rate(0.5);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdaBoostParams<P, R>(AdaBoostValidParams<P, R>);

impl<P> AdaBoostParams<P, ThreadRng> {
    /// Create a new AdaBoost parameter set with default values and a thread-local RNG.
    ///
    /// # Arguments
    ///
    /// * `model_params` - The parameters for the base learner (e.g., DecisionTreeParams)
    ///
    /// # Default Values
    ///
    /// * `n_estimators`: 50
    /// * `learning_rate`: 1.0
    pub fn new(model_params: P) -> AdaBoostParams<P, ThreadRng> {
        Self::new_fixed_rng(model_params, rand::thread_rng())
    }
}

impl<P, R: Rng + Clone> AdaBoostParams<P, R> {
    /// Create a new AdaBoost parameter set with a fixed RNG for reproducibility.
    ///
    /// # Arguments
    ///
    /// * `model_params` - The parameters for the base learner
    /// * `rng` - A seeded random number generator for reproducible results
    ///
    /// # Example
    ///
    /// ```no_run
    /// use linfa_ensemble::AdaBoostParams;
    /// use linfa_trees::DecisionTree;
    /// use ndarray_rand::rand::SeedableRng;
    /// use rand::rngs::SmallRng;
    ///
    /// let rng = SmallRng::seed_from_u64(42);
    /// let params = AdaBoostParams::new_fixed_rng(
    ///     DecisionTree::<f64, usize>::params().max_depth(Some(1)),
    ///     rng
    /// );
    /// ```
    pub fn new_fixed_rng(model_params: P, rng: R) -> AdaBoostParams<P, R> {
        Self(AdaBoostValidParams {
            n_estimators: 50,
            learning_rate: 1.0,
            model_params,
            rng,
        })
    }

    /// Set the maximum number of weak learners to train.
    ///
    /// # Arguments
    ///
    /// * `n_estimators` - Must be at least 1. Typical values: 50-500
    ///
    /// # Notes
    ///
    /// Higher values generally lead to better training performance but:
    /// * Increase training time linearly
    /// * May lead to overfitting
    /// * Should be balanced with `learning_rate`
    pub fn n_estimators(mut self, n_estimators: usize) -> Self {
        self.0.n_estimators = n_estimators;
        self
    }

    /// Set the learning rate (shrinkage parameter).
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - Must be positive. Typical values: 0.01 to 2.0
    ///
    /// # Notes
    ///
    /// * Values < 1.0 provide regularization and often improve generalization
    /// * Lower values require more estimators to achieve similar performance
    /// * A common strategy is to use learning_rate=0.1 with n_estimators=500
    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.0.learning_rate = learning_rate;
        self
    }
}

impl<P, R> ParamGuard for AdaBoostParams<P, R> {
    type Checked = AdaBoostValidParams<P, R>;
    type Error = Error;

    fn check_ref(&self) -> Result<&Self::Checked> {
        if self.0.n_estimators < 1 {
            Err(Error::Parameters(format!(
                "n_estimators must be at least 1, but was {}",
                self.0.n_estimators
            )))
        } else if self.0.learning_rate <= 0.0 {
            Err(Error::Parameters(format!(
                "learning_rate must be positive, but was {}",
                self.0.learning_rate
            )))
        } else if !self.0.learning_rate.is_finite() {
            Err(Error::Parameters(
                "learning_rate must be finite (not NaN or infinity)".to_string(),
            ))
        } else {
            Ok(&self.0)
        }
    }

    fn check(self) -> Result<Self::Checked> {
        self.check_ref()?;
        Ok(self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use linfa_trees::DecisionTree;
    use ndarray_rand::rand::SeedableRng;
    use rand::rngs::SmallRng;

    #[test]
    fn test_default_params() {
        let rng = SmallRng::seed_from_u64(42);
        let params = AdaBoostParams::new_fixed_rng(DecisionTree::<f64, usize>::params(), rng);
        assert_eq!(params.0.n_estimators, 50);
        assert_eq!(params.0.learning_rate, 1.0);
    }

    #[test]
    fn test_custom_params() {
        let rng = SmallRng::seed_from_u64(42);
        let params = AdaBoostParams::new_fixed_rng(DecisionTree::<f64, usize>::params(), rng)
            .n_estimators(100)
            .learning_rate(0.5);
        assert_eq!(params.0.n_estimators, 100);
        assert_eq!(params.0.learning_rate, 0.5);
    }

    #[test]
    fn test_invalid_n_estimators() {
        let rng = SmallRng::seed_from_u64(42);
        let params = AdaBoostParams::new_fixed_rng(DecisionTree::<f64, usize>::params(), rng)
            .n_estimators(0);
        assert!(params.check_ref().is_err());
    }

    #[test]
    fn test_invalid_learning_rate_negative() {
        let rng = SmallRng::seed_from_u64(42);
        let params = AdaBoostParams::new_fixed_rng(DecisionTree::<f64, usize>::params(), rng)
            .learning_rate(-0.5);
        assert!(params.check_ref().is_err());
    }

    #[test]
    fn test_invalid_learning_rate_zero() {
        let rng = SmallRng::seed_from_u64(42);
        let params = AdaBoostParams::new_fixed_rng(DecisionTree::<f64, usize>::params(), rng)
            .learning_rate(0.0);
        assert!(params.check_ref().is_err());
    }

    #[test]
    fn test_invalid_learning_rate_nan() {
        let rng = SmallRng::seed_from_u64(42);
        let params = AdaBoostParams::new_fixed_rng(DecisionTree::<f64, usize>::params(), rng)
            .learning_rate(f64::NAN);
        assert!(params.check_ref().is_err());
    }

    #[test]
    fn test_valid_params() {
        let rng = SmallRng::seed_from_u64(42);
        let params = AdaBoostParams::new_fixed_rng(DecisionTree::<f64, usize>::params(), rng)
            .n_estimators(100)
            .learning_rate(0.5);
        assert!(params.check_ref().is_ok());
    }
}
