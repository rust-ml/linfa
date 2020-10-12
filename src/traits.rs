//! Provide traits for different classes of algorithms
//!

use crate::dataset::{Data, Dataset};

/// Transformation algorithms
///
/// A transformer takes a dataset and transforms it into a different one. It has no concept of
/// state and provides therefore no method to predict new data. A typical example are kernel
/// methods.
///
/// It should be implemented for all algorithms, also for those which can be fitted.
///
pub trait Transformer<D: Data, T> {
    fn transform(&self, x: D) -> T;
}

/// Fittable algorithms
///
/// A fittable algorithm takes a dataset and creates a concept of some kind about it. For example
/// in *KMeans* this would be the mean values for each class, or in *SVM* the separating
/// hyperplane. It returns a model, which can be used to predict targets for new data.
pub trait Fit<D: Data, T> {
    type Object: Predict<D, T>;

    fn fit(&self, dataset: Dataset<D, T>) -> Self::Object;
}

/// Incremental algorithms
///
/// An incremental algorithm takes a former model and dataset and returns a new model with updated
/// parameters. If the former model is `None`, then the function acts like `Fit::fit` and
/// initializes the model first.
pub trait IncrementalFit<D: Data, T> {
    type Object: Predict<D, T>;

    fn fit_with<I: Into<Option<Self::Object>>>(&self, model: I, dataset: Dataset<D, T>) -> Self::Object;
}

/// Predict with model
pub trait Predict<D: Data, T> {
    fn predict(&self, x: D) -> T;
}
