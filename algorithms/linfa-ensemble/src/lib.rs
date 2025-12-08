//! # Ensemble Learning Algorithms
//!
//! Ensemble methods combine the predictions of several base estimators built with a given
//! learning algorithm in order to improve generalizability / robustness over a single estimator.
//!
//! This crate (`linfa-ensemble`), provides pure Rust implementations of popular ensemble techniques, such as
//! * [Boostrap Aggregation](EnsembleLearner)
//! * [Random Forest](RandomForest)
//! * [AdaBoost]
//!
//! ## Bootstrap Aggregation (aka Bagging)
//!
//! A typical example of ensemble method is Bootstrap Aggregation, which combines the predictions of
//! several decision trees (see [`linfa-trees`](linfa_trees)) trained on different samples subset of the training dataset.
//!
//! ## Random Forest
//!
//! A special case of Bootstrap Aggregation using decision trees (see  [`linfa-trees`](linfa_trees)) with random feature
//! selection. A typical number of random prediction to be selected is $\sqrt{p}$ with $p$ being
//! the number of available features.
//!
//! ## AdaBoost
//!
//! AdaBoost (Adaptive Boosting) is a boosting ensemble method that trains weak learners sequentially.
//! Each subsequent learner focuses on the examples that previous learners misclassified by increasing
//! their sample weights. The final prediction is a weighted vote of all learners, where better-performing
//! learners receive higher weights. Unlike bagging methods, boosting creates a strong classifier from
//! weak learners (typically shallow decision trees or "stumps").
//!
//! ## Reference
//!
//! * [Scikit-Learn User Guide](https://scikit-learn.org/stable/modules/ensemble.html)
//! * [An Introduction to Statistical Learning](https://www.statlearning.com/)
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
//!     .ensemble_size(100)        // Number of Decision Tree to fit
//!     .bootstrap_proportion(0.7) // Select only 70% of the data via bootstrap
//!     .fit(&train)
//!     .unwrap();
//!
//! // Make predictions on the test set
//! let predictions = bagging_model.predict(&test);
//! ```
//!
//! This example shows how to train a [Random Forest](RandomForest) model using 100 decision trees,
//! each trained on 70% of the training data (bootstrap sampling) and using only
//! 30% of the available features.
//!
//! ```no_run
//! use linfa::prelude::{Fit, Predict};
//! use linfa_ensemble::RandomForestParams;
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
//! let random_forest = RandomForestParams::new(DecisionTree::params())
//!     .ensemble_size(100)        // Number of Decision Tree to fit
//!     .bootstrap_proportion(0.7) // Select only 70% of the data via bootstrap
//!     .feature_proportion(0.3)   // Select only 30% of the feature
//!     .fit(&train)
//!     .unwrap();
//!
//! // Make predictions on the test set
//! let predictions = random_forest.predict(&test);
//! ```

mod adaboost;
mod adaboost_hyperparams;
mod algorithm;
mod hyperparams;

pub use adaboost::*;
pub use adaboost_hyperparams::*;
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

    #[test]
    fn test_adaboost_accuracy_on_iris_dataset() {
        let mut rng = SmallRng::seed_from_u64(42);
        let (train, test) = linfa_datasets::iris()
            .shuffle(&mut rng)
            .split_with_ratio(0.8);

        // Train AdaBoost with decision tree stumps (shallow trees)
        let model = AdaBoostParams::new_fixed_rng(DecisionTree::params().max_depth(Some(1)), rng)
            .n_estimators(50)
            .learning_rate(1.0)
            .fit(&train)
            .unwrap();

        let predictions = model.predict(&test);

        let cm = predictions.confusion_matrix(&test).unwrap();
        let acc = cm.accuracy();
        assert!(
            acc >= 0.85,
            "Expected accuracy to be above 85%, got {}",
            acc
        );
    }

    #[test]
    fn test_adaboost_with_low_learning_rate() {
        let mut rng = SmallRng::seed_from_u64(42);
        let (train, test) = linfa_datasets::iris()
            .shuffle(&mut rng)
            .split_with_ratio(0.8);

        // Train AdaBoost with lower learning rate and more estimators
        let model = AdaBoostParams::new_fixed_rng(DecisionTree::params().max_depth(Some(2)), rng)
            .n_estimators(100)
            .learning_rate(0.5)
            .fit(&train)
            .unwrap();

        let predictions = model.predict(&test);

        let cm = predictions.confusion_matrix(&test).unwrap();
        let acc = cm.accuracy();
        assert!(
            acc >= 0.85,
            "Expected accuracy to be above 85%, got {}",
            acc
        );
    }

    #[test]
    fn test_adaboost_model_weights() {
        let mut rng = SmallRng::seed_from_u64(42);
        let (train, _) = linfa_datasets::iris()
            .shuffle(&mut rng)
            .split_with_ratio(0.8);

        let model = AdaBoostParams::new_fixed_rng(DecisionTree::params().max_depth(Some(1)), rng)
            .n_estimators(10)
            .fit(&train)
            .unwrap();

        // Verify that model weights are positive
        for weight in model.weights() {
            assert!(
                *weight > 0.0,
                "Model weight should be positive, got {}",
                weight
            );
        }

        // Verify we have the expected number of models
        assert_eq!(model.n_estimators(), 10);
    }
}
