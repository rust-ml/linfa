//! ## Ensemble Learning Algorithms
//!
//! Ensemble methods combine the predictions of several base estimators built with a given
//! learning algorithm in order to improve generalizability / robustness over a single estimator.
//!
//! ## Random Forest
//!
//! Typical example of ensemble method is random forest, which combines the predictions of
//! several decision trees trained on different parts of the same training set.
//!
//! ### Example
//!
//! This example shows how to train a random forest model using 100 decision trees,
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
//! // Train a random forest model on the iris dataset
//! let random_forest_model = EnsembleLearnerParams::new(DecisionTree::params())
//!     .ensemble_size(100)
//!     .bootstrap_proportion(0.7)
//!     .fit(&train)
//!     .unwrap();
//!
//! // Make predictions on the test set
//! let predictions = random_forest_model.predict(&test);
//! ```
//!
mod algorithm;
mod hyperparams;

pub use algorithm::*;
pub use hyperparams::*;
