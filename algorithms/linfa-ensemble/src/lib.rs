//! # Ensemble Learning Algorithms
//!
//! Ensemble methods combine the predictions of several base estimators built with a given
//! learning algorithm in order to improve generalizability / robustness over a single estimator.
//!
//! ## Bootstrap Aggregation (aka Bagging)
//!
//! A typical example of ensemble method is Bootstrapo AGgregation, which combines the predictions of
//! several decision trees (see `linfa-trees`) trained on different samples subset of the training dataset.
//!
//! ## Reference
//!
//! * [Scikit-Learn User Guide](https://scikit-learn.org/stable/modules/ensemble.html)
//!
//! ## Example
//!
//! This example shows how to train a bagging model using 100 decision trees,
//! each trained on 70% of the training data (bootstrap sampling).
//!
//! ```no_run
//! use linfa::prelude::{Fit, Predict};
//! use linfa_ensemble::EnsembleLearnerParams;
//! use linfa_trees::DecisionTree;
//! use ndarray_rand::rand::SeedableRng;
//! use rand::rngs::SmallRng;
//!
//! // Load Iris dataset
//! let mut rng = SmallRng::seed_from_u64(42);
//! let (train, test) = linfa_datasets::iris()
//!     .shuffle(&mut rng)
//!     .split_with_ratio(0.8);
//!
//! // Train the model on the iris dataset
//! let bagging_model = EnsembleLearnerParams::new(DecisionTree::params())
//!     .ensemble_size(100)
//!     .bootstrap_proportion(0.7)
//!     .fit(&train)
//!     .unwrap();
//!
//! // Make predictions on the test set
//! let predictions = bagging_model.predict(&test);
//! ```
//!
mod algorithm;
mod hyperparams;

pub use algorithm::*;
pub use hyperparams::*;

#[cfg(test)]
mod tests {
    use super::*;
    use linfa::prelude::{Fit, Predict, ToConfusionMatrix};
    use linfa_trees::DecisionTree;
    use ndarray_rand::rand::SeedableRng;
    use rand::rngs::SmallRng;

    #[test]
    fn test_random_forest_accuracy_on_iris_dataset() {
        let mut rng = SmallRng::seed_from_u64(42);
        let (train, test) = linfa_datasets::iris()
            .shuffle(&mut rng)
            .split_with_ratio(0.8);

        let model = RandomForestParams::new_fixed_rng(DecisionTree::params(), rng)
            .ensemble_size(100)
            .bootstrap_proportion(0.7)
            .feature_proportion(0.3)
            .fit(&train)
            .unwrap();

        let predictions = model.predict(&test);

        let cm = predictions.confusion_matrix(&test).unwrap();
        let acc = cm.accuracy();
        assert!(acc >= 0.9, "Expected accuracy to be above 90%, got {}", acc);
    }

    #[test]
    fn test_ensemble_learner_accuracy_on_iris_dataset() {
        let mut rng = SmallRng::seed_from_u64(42);
        let (train, test) = linfa_datasets::iris()
            .shuffle(&mut rng)
            .split_with_ratio(0.8);

        let model = EnsembleLearnerParams::new_fixed_rng(DecisionTree::params(), rng)
            .ensemble_size(100)
            .bootstrap_proportion(0.7)
            .fit(&train)
            .unwrap();

        let predictions = model.predict(&test);

        let cm = predictions.confusion_matrix(&test).unwrap();
        let acc = cm.accuracy();
        assert!(acc >= 0.9, "Expected accuracy to be above 90%, got {}", acc);
    }

}
