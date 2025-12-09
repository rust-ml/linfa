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

    #[test]
    fn test_adaboost_early_stopping_on_perfect_fit() {
        use ndarray::Array2;
        use linfa::DatasetBase;

        // Create a simple linearly separable dataset
        let records = Array2::from_shape_vec(
            (6, 2),
            vec![
                0.0, 0.0, // class 0
                0.1, 0.1, // class 0
                0.2, 0.2, // class 0
                1.0, 1.0, // class 1
                1.1, 1.1, // class 1
                1.2, 1.2, // class 1
            ],
        )
        .unwrap();
        let targets = ndarray::array![0, 0, 0, 1, 1, 1];
        let dataset = DatasetBase::new(records, targets);

        let rng = SmallRng::seed_from_u64(42);
        let model = AdaBoostParams::new_fixed_rng(DecisionTree::params().max_depth(Some(3)), rng)
            .n_estimators(50)
            .fit(&dataset)
            .unwrap();

        // Should stop early due to perfect classification
        assert!(
            model.n_estimators() < 50,
            "Expected early stopping, but got {} estimators",
            model.n_estimators()
        );
    }

    #[test]
    fn test_adaboost_single_class_error() {
        use ndarray::Array2;
        use linfa::DatasetBase;

        // Create dataset with only one class
        let records = Array2::from_shape_vec(
            (4, 2),
            vec![0.0, 0.0, 0.1, 0.1, 0.2, 0.2, 0.3, 0.3],
        )
        .unwrap();
        let targets = ndarray::array![0, 0, 0, 0]; // All same class
        let dataset = DatasetBase::new(records, targets);

        let rng = SmallRng::seed_from_u64(42);
        let result = AdaBoostParams::new_fixed_rng(DecisionTree::params(), rng)
            .n_estimators(10)
            .fit(&dataset);

        assert!(
            result.is_err(),
            "Should fail with single class dataset"
        );
    }

    #[test]
    fn test_adaboost_classes_method() {
        let mut rng = SmallRng::seed_from_u64(42);
        let (train, _) = linfa_datasets::iris()
            .shuffle(&mut rng)
            .split_with_ratio(0.8);

        let model = AdaBoostParams::new_fixed_rng(DecisionTree::params().max_depth(Some(1)), rng)
            .n_estimators(10)
            .fit(&train)
            .unwrap();

        // Verify classes are properly stored
        let classes = &model.classes;
        assert_eq!(classes.len(), 3, "Iris has 3 classes");
        assert_eq!(classes, &vec![0, 1, 2], "Classes should be [0, 1, 2]");
    }
}
