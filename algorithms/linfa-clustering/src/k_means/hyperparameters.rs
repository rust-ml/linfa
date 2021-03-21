use super::init::KMeansInit;
use linfa::Float;
use ndarray_rand::rand::{Rng, SeedableRng};
use rand_isaac::Isaac64Rng;
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
    n_runs: u64,
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
    init: KMeansInit,
    /// The random number generator
    rng: R,
}

/// An helper struct used to construct a set of [valid hyperparameters](struct.KMeansHyperParams.html) for
/// the [K-means algorithm](struct.KMeans.html) (using the builder pattern).
pub struct KMeansHyperParamsBuilder<F: Float, R: Rng> {
    n_runs: u64,
    tolerance: F,
    max_n_iterations: u64,
    n_clusters: usize,
    init: KMeansInit,
    rng: R,
}

impl<F: Float, R: Rng + Clone> KMeansHyperParamsBuilder<F, R> {
    /// Set the value of `n_runs`.
    ///
    /// The final results will be the best output of n_runs consecutive runs in terms of inertia
    /// (sum of squared distances to the closest centroid for all observations in the training set)
    pub fn n_runs(mut self, n_runs: u64) -> Self {
        self.n_runs = n_runs;
        self
    }

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
    pub fn tolerance(mut self, tolerance: F) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Set the value of `init`.
    ///
    /// Before training, the centroids are initialized using the method specified by `init`.
    /// Currently the choices for initialization are `Random` and `KMeansPP`.
    pub fn init_method(mut self, init: KMeansInit) -> Self {
        self.init = init;
        self
    }

    /// Return an instance of `KMeansHyperParams` after
    /// having performed validation checks on all the specified hyperparameters.
    ///
    /// **Panics** if any of the validation checks fails.
    pub fn build(&self) -> KMeansHyperParams<F, R> {
        KMeansHyperParams::build(
            self.n_clusters,
            self.n_runs,
            self.tolerance,
            self.max_n_iterations,
            self.init,
            self.rng.clone(),
        )
    }
}

impl<F: Float> KMeansHyperParams<F, Isaac64Rng> {
    pub fn new(n_clusters: usize) -> KMeansHyperParamsBuilder<F, Isaac64Rng> {
        Self::new_with_rng(n_clusters, Isaac64Rng::seed_from_u64(42))
    }
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
    /// * `tolerance = 1e-4`;
    /// * `max_n_iterations = 300`.

    pub fn new_with_rng(n_clusters: usize, rng: R) -> KMeansHyperParamsBuilder<F, R> {
        KMeansHyperParamsBuilder {
            n_runs: 10,
            tolerance: F::from(1e-4).unwrap(),
            max_n_iterations: 300,
            n_clusters,
            init: KMeansInit::Random,
            rng,
        }
    }

    /// The final results will be the best output of n_runs consecutive runs in terms of inertia.
    pub fn n_runs(&self) -> u64 {
        self.n_runs
    }

    /// The training is considered complete if the euclidean distance
    /// between the old set of centroids and the new set of centroids
    /// after a training iteration is lower or equal than `tolerance`.
    pub fn tolerance(&self) -> F {
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

    /// Cluster initialization strategy
    pub fn init(&self) -> KMeansInit {
        self.init
    }

    /// Returns a clone of the random generator
    pub fn rng(&self) -> R {
        self.rng.clone()
    }

    fn build(
        n_clusters: usize,
        n_runs: u64,
        tolerance: F,
        max_n_iterations: u64,
        init: KMeansInit,
        rng: R,
    ) -> Self {
        if n_runs == 0 {
            panic!("`n_runs` cannot be 0!");
        }
        if max_n_iterations == 0 {
            panic!("`max_n_iterations` cannot be 0!");
        }
        if tolerance <= F::zero() {
            panic!("`tolerance` must be greater than 0!");
        }
        if n_clusters == 0 {
            panic!("`n_clusters` cannot be 0!");
        }
        KMeansHyperParams {
            n_runs,
            tolerance,
            max_n_iterations,
            n_clusters,
            init,
            rng,
        }
    }
}
