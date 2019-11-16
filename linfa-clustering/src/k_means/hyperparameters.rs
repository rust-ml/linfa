use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
/// The set of hyperparameters that can be specified for the execution of
/// the [K-means algorithm](struct.KMeans.html).
pub struct KMeansHyperParams {
    /// The training is considered complete if the euclidean distance
    /// between the old set of centroids and the new set of centroids
    /// after a training iteration is lower or equal than `tolerance`.
    tolerance: f64,
    /// We exit the training loop when the number of training iterations
    /// exceeds `max_n_iterations` even if the `tolerance` convergence
    /// condition has not been met.
    max_n_iterations: u64,
    /// The number of clusters we will be looking for in the training dataset.
    n_clusters: usize,
}

/// An helper struct used to construct a set of [valid hyperparameters](struct.KMeansHyperParams.html) for
/// the [K-means algorithm](struct.KMeans.html) (using the builder pattern).
pub struct KMeansHyperParamsBuilder {
    tolerance: f64,
    max_n_iterations: u64,
    n_clusters: usize,
}

impl KMeansHyperParamsBuilder {
    /// Set the value of `max_n_iterations`.
    ///
    /// We exit the training loop when the number of training iterations
    /// exceeds `max_n_iterations` even if the `tolerance` convergence
    /// condition has not been met.
    pub fn max_n_iterations(mut self, max_n_iterations: u64) -> Self {
        self.max_n_iterations = max_n_iterations;
        self
    }

    /// Set the value of `tolerance`.
    ///
    /// The training is considered complete if the euclidean distance
    /// between the old set of centroids and the new set of centroids
    /// after a training iteration is lower or equal than `tolerance`.
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Return an instance of `KMeansHyperParams` after
    /// having performed validation checks on all the specified hyperparamters.
    ///
    /// **Panics** if any of the validation checks fails.
    pub fn build(self) -> KMeansHyperParams {
        KMeansHyperParams::build(self.n_clusters, self.tolerance, self.max_n_iterations)
    }
}

impl KMeansHyperParams {
    /// `new` lets us configure our training algorithm parameters:
    /// * we will be looking for `n_clusters` in the training dataset;
    /// * the training is considered complete if the euclidean distance
    ///   between the old set of centroids and the new set of centroids
    ///   after a training iteration is lower or equal than `tolerance`;
    /// * we exit the training loop when the number of training iterations
    ///   exceeds `max_n_iterations` even if the `tolerance` convergence
    ///   condition has not been met.
    ///
    /// `n_clusters` is mandatory.
    ///
    /// Defaults are provided if optional parameters are not specified:
    /// * `tolerance = 1e-4`;
    /// * `max_n_iterations = 300`.
    pub fn new(n_clusters: usize) -> KMeansHyperParamsBuilder {
        KMeansHyperParamsBuilder {
            tolerance: 1e-4,
            max_n_iterations: 300,
            n_clusters,
        }
    }

    /// The training is considered complete if the euclidean distance
    /// between the old set of centroids and the new set of centroids
    /// after a training iteration is lower or equal than `tolerance`.
    pub fn tolerance(&self) -> f64 {
        self.tolerance
    }

    /// We exit the training loop when the number of training iterations
    /// exceeds `max_n_iterations` even if the `tolerance` convergence
    /// condition has not been met.
    pub fn max_n_iterations(&self) -> u64 {
        self.max_n_iterations
    }

    /// The number of clusters we will be looking for in the training dataset.
    pub fn n_clusters(&self) -> usize {
        self.n_clusters
    }

    fn build(n_clusters: usize, tolerance: f64, max_n_iterations: u64) -> Self {
        if max_n_iterations == 0 {
            panic!("`max_n_iterations` cannot be 0!");
        }
        if tolerance <= 0. {
            panic!("`tolerance` must be greater than 0!");
        }
        if n_clusters == 0 {
            panic!("`n_clusters` cannot be 0!");
        }
        KMeansHyperParams {
            tolerance,
            max_n_iterations,
            n_clusters,
        }
    }
}
