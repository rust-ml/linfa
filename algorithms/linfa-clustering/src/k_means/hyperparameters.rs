use super::init::KMeansInit;
use linfa::Float;
use ndarray_rand::rand::Rng;
#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};

#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Clone, Debug, PartialEq)]
/// The set of hyperparameters that can be specified for the execution of
/// the [K-means algorithm](struct.KMeans.html).
pub struct KMeansHyperParams<F: Float, R: Rng> {
    /// Number of time the k-means algorithm will be run with different centroid seeds.
    n_runs: usize,
    /// The training is considered complete if the euclidean distance
    /// between the old set of centroids and the new set of centroids
    /// after a training iteration is lower or equal than `tolerance`.
    tolerance: F,
    /// We exit the training loop when the number of training iterations
    /// exceeds `max_n_iterations` even if the `tolerance` convergence
    /// condition has not been met.
    max_n_iterations: u64,
    /// The number of clusters we will be looking for in the training dataset.
    n_clusters: usize,
    /// The initialization strategy used to initialize the centroids.
    init: KMeansInit<F>,
    /// The random number generator
    rng: R,
}

impl<F: Float, R: Rng + Clone> KMeansHyperParams<F, R> {
    /// `new` lets us configure our training algorithm parameters:
    /// * we will be looking for `n_clusters` in the training dataset;
    /// * the training is considered complete if the euclidean distance
    ///   between the old set of centroids and the new set of centroids
    ///   after a training iteration is lower or equal than `tolerance`;
    /// * we exit the training loop when the number of training iterations
    ///   exceeds `max_n_iterations` even if the `tolerance` convergence
    ///   condition has not been met.
    /// * As KMeans convergence depends on centroids initialization
    ///   we run the algorithm `n_runs` times and we keep the best outputs
    ///   in terms of inertia that the ones which minimizes the sum of squared
    ///   euclidean distances to the closest centroid for all observations.
    ///
    /// `n_clusters` is mandatory.
    ///
    /// Defaults are provided if optional parameters are not specified:
    /// * `tolerance = 1e-4`
    /// * `max_n_iterations = 300`
    /// * `n_runs = 10`
    /// * `init = KMeansPlusPlus`

    pub fn new_with_rng(n_clusters: usize, rng: R) -> Self {
        if n_clusters == 0 {
            panic!("`n_clusters` cannot be 0!");
        }
        Self {
            n_runs: 10,
            tolerance: F::cast(1e-4),
            max_n_iterations: 300,
            n_clusters,
            init: KMeansInit::KMeansPlusPlus,
            rng,
        }
    }

    /// The final results will be the best output of n_runs consecutive runs in terms of inertia.
    pub fn get_n_runs(&self) -> usize {
        self.n_runs
    }

    /// The training is considered complete if the euclidean distance
    /// between the old set of centroids and the new set of centroids
    /// after a training iteration is lower or equal than `tolerance`.
    pub fn get_tolerance(&self) -> F {
        self.tolerance
    }

    /// We exit the training loop when the number of training iterations
    /// exceeds `max_n_iterations` even if the `tolerance` convergence
    /// condition has not been met.
    pub fn get_max_n_iterations(&self) -> u64 {
        self.max_n_iterations
    }

    /// The number of clusters we will be looking for in the training dataset.
    pub fn get_n_clusters(&self) -> usize {
        self.n_clusters
    }

    /// Cluster initialization strategy
    pub fn get_init_method(&self) -> &KMeansInit<F> {
        &self.init
    }

    /// Returns a clone of the random generator
    pub fn get_rng(&self) -> R {
        self.rng.clone()
    }

    /// Change the value of `n_runs`
    pub fn n_runs(mut self, n_runs: usize) -> Self {
        if n_runs == 0 {
            panic!("`n_runs` cannot be 0!");
        }
        self.n_runs = n_runs;
        self
    }

    /// Change the value of `tolerance`
    pub fn tolerance(mut self, tolerance: F) -> Self {
        if tolerance <= F::zero() {
            panic!("`tolerance` must be greater than 0!");
        }
        self.tolerance = tolerance;
        self
    }

    /// Change the value of `max_n_iterations`
    pub fn max_n_iterations(mut self, max_n_iterations: u64) -> Self {
        if max_n_iterations == 0 {
            panic!("`max_n_iterations` cannot be 0!");
        }
        self.max_n_iterations = max_n_iterations;
        self
    }

    /// Change the value of `init`
    pub fn init_method(mut self, init: KMeansInit<F>) -> Self {
        self.init = init;
        self
    }
}

#[cfg(test)]
mod tests {
    use crate::KMeans;

    #[test]
    #[should_panic]
    fn n_clusters_cannot_be_zero() {
        KMeans::<f32>::params(0);
    }

    #[test]
    #[should_panic]
    fn tolerance_has_to_positive() {
        KMeans::params(1).tolerance(-1.);
    }

    #[test]
    #[should_panic]
    fn tolerance_cannot_be_zero() {
        KMeans::params(1).tolerance(0.);
    }

    #[test]
    #[should_panic]
    fn max_n_iterations_cannot_be_zero() {
        KMeans::params(1).tolerance(1.).max_n_iterations(0);
    }

    #[test]
    #[should_panic]
    fn n_runs_cannot_be_zero() {
        KMeans::params(1).tolerance(1.).n_runs(0);
    }
}
