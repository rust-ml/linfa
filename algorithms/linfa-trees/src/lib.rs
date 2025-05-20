//!
//! # Decision tree learning
//! `linfa-trees` aims to provide pure Rust implementations
//! of decision tree learning algorithms.
//!
//! # The big picture
//!
//! `linfa-trees` is a crate in the [linfa](https://github.com/rust-ml/linfa) ecosystem,
//! an effort to create a toolkit for classical Machine Learning implemented in pure Rust, akin to Python's scikit-learn.
//!
//! Decision Trees (DTs) are a non-parametric supervised learning method used for classification and regression.
//! The goal is to create a model that predicts the value of a target variable by learning simple decision rules
//! inferred from the data features.
//!
//! # Current state
//!
//! `linfa-trees` currently provides an [implementation](DecisionTree) of single-tree fitting for classification.
//!

mod decision_trees;

// Re-export all core decision tree functionality
pub use decision_trees::*;

// Explicitly export the Random Forest classifier API
pub use decision_trees::random_forest::{RandomForestClassifier, RandomForestParams};

// Re-export the common Result alias for convenience
pub use linfa::error::Result;
