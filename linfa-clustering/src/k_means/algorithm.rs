use crate::k_means::hyperparameters::KMeansHyperParams;
use ndarray::{s, Array1, Array2, ArrayBase, Axis, Data, DataMut, Ix1, Ix2, Zip};
use ndarray_rand::rand;
use ndarray_rand::rand::Rng;
use ndarray_stats::DeviationExt;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
/// K-means clustering aims to partition a set of observations into clusters,
/// where each observation belongs to the cluster with the nearest mean.
///
/// The mean of the points within a cluster is called *centroid*.
///
/// Given the set of centroids, you can assign an observation to a cluster
/// choosing the nearest centroid.
///
/// We provide an implementation of the _standard algorithm_, also known as
/// Lloyd's algorithm or naive K-means.
///
/// More details on the algorithm can be found in the next section or
/// [here](https://en.wikipedia.org/wiki/K-means_clustering).
///
/// ## The algorithm
///
/// K-means is an iterative algorithm: it progressively refines the choice of centroids.
///
/// It's guaranteed to converge, even though it might not find the optimal set of centroids
/// (unfortunately it can get stuck in a local minimum, finding the optimal minimum if NP-hard!).
///
/// There are three steps in the standard algorithm:
/// - initialisation step: how do we choose our initial set of centroids?
/// - assignment step: assign each observation to the nearest cluster
///                    (minimum distance between the observation and the cluster's centroid);
/// - update step: recompute the centroid of each cluster.
///
/// The initialisation step is a one-off, done at the very beginning.
/// Assignment and update are repeated in a loop until convergence is reached (either the
/// euclidean distance between the old and the new clusters is below `tolerance` or
/// we exceed the `max_n_iterations`).
///
/// ## Parallelisation
///
/// The work performed by the assignment step does not require any coordination:
/// the closest centroid for each point can be computed independently from the
/// closest centroid for any of the remaining points.
///
/// This makes it a good candidate for parallel execution: `KMeans::fit` parallelises the
/// assignment step thanks to the `rayon` feature in `ndarray`.
///
/// The update step requires a bit more coordination (computing a rolling mean in
/// parallel) but it is still parallelisable.
/// Nonetheless, our first attempts have not improved performance
/// (most likely due to our strategy used to split work between threads), hence
/// the update step is currently executed on a single thread.
///
/// ## Tutorial
///
/// Let's do a walkthrough of a training-predict-save example.
///
/// ```
/// use linfa_clustering::{KMeansHyperParams, KMeans, generate_blobs};
/// use ndarray::{Axis, array, s};
/// use ndarray_rand::rand::SeedableRng;
/// use rand_isaac::Isaac64Rng;
/// use approx::assert_abs_diff_eq;
///
/// // Our random number generator, seeded for reproducibility
/// let seed = 42;
/// let mut rng = Isaac64Rng::seed_from_u64(seed);
///
/// // `expected_centroids` has shape `(n_centroids, n_features)`
/// // i.e. three points in the 2-dimensional plane
/// let expected_centroids = array![[0., 1.], [-10., 20.], [-1., 10.]];
/// // Let's generate a synthetic dataset: three blobs of observations
/// // (100 points each) centered around our `expected_centroids`
/// let observations = generate_blobs(100, &expected_centroids, &mut rng);
///
/// // Let's configure and run our K-means algorithm
/// // We use the builder pattern to specify the hyperparameters
/// // `n_clusters` is the only mandatory parameter.
/// // If you don't specify the others (e.g. `tolerance` or `max_n_iterations`)
/// // default values will be used.
/// let n_clusters = expected_centroids.len_of(Axis(0));
/// let hyperparams = KMeansHyperParams::new(n_clusters)
///     .tolerance(1e-2)
///     .build();
/// // Let's run the algorithm!
/// let model = KMeans::fit(hyperparams, &observations, &mut rng);
///
/// // Once we found our set of centroids, we can also assign new points to the nearest cluster
/// let new_observation = array![[-9., 20.5]];
/// // Predict returns the **index** of the nearest cluster
/// let closest_cluster_index = model.predict(&new_observation);
/// // We can retrieve the actual centroid of the closest cluster using `.centroids()`
/// let closest_centroid = &model.centroids().index_axis(Axis(0), closest_cluster_index[0]);
///
/// // The model can be serialised (and deserialised) to disk using serde
/// // We'll use the JSON format here for simplicity
/// let filename = "k_means_model.json";
/// let writer = std::fs::File::create(filename).expect("Failed to open file.");
/// serde_json::to_writer(writer, &model).expect("Failed to serialise model.");
///
/// let reader = std::fs::File::open(filename).expect("Failed to open file.");
/// let loaded_model: KMeans = serde_json::from_reader(reader).expect("Failed to deserialise model");
///
/// assert_abs_diff_eq!(model.centroids(), loaded_model.centroids(), epsilon = 1e-10);
/// assert_eq!(model.hyperparameters(), loaded_model.hyperparameters());
/// ```
///
pub struct KMeans {
    hyperparameters: KMeansHyperParams,
    centroids: Array2<f64>,
}

impl KMeans {
    /// Given an input matrix `observations`, with shape `(n_observations, n_features)`,
    /// `fit` identifies `n_clusters` centroids based on the training data distribution.
    ///
    /// An instance of `KMeans` is returned.
    ///
    pub fn fit(
        hyperparameters: KMeansHyperParams,
        observations: &ArrayBase<impl Data<Elem = f64> + Sync, Ix2>,
        rng: &mut impl Rng,
    ) -> Self {
        let mut centroids = get_random_centroids(hyperparameters.n_clusters(), observations, rng);

        let mut has_converged;
        let mut n_iterations = 0;

        let mut memberships = Array1::zeros(observations.dim().0);

        loop {
            update_cluster_memberships(&centroids, observations, &mut memberships);
            let new_centroids =
                compute_centroids(hyperparameters.n_clusters(), observations, &memberships);

            let distance = centroids
                .sq_l2_dist(&new_centroids)
                .expect("Failed to compute distance");
            has_converged = distance < hyperparameters.tolerance()
                || n_iterations > hyperparameters.max_n_iterations();

            centroids = new_centroids;
            n_iterations += 1;

            if has_converged {
                break;
            }
        }

        Self {
            hyperparameters,
            centroids,
        }
    }

    /// Given an input matrix `observations`, with shape `(n_observations, n_features)`,
    /// `predict` returns, for each observation, the index of the closest cluster/centroid.
    ///
    /// You can retrieve the centroid associated to an index using the
    /// [`centroids` method](#method.centroids).
    pub fn predict(&self, observations: &ArrayBase<impl Data<Elem = f64>, Ix2>) -> Array1<usize> {
        compute_cluster_memberships(&self.centroids, observations)
    }

    /// Return the set of centroids as a 2-dimensional matrix with shape
    /// `(n_centroids, n_features)`.
    pub fn centroids(&self) -> &Array2<f64> {
        &self.centroids
    }

    /// Return the hyperparameters used to train this K-means model instance.
    pub fn hyperparameters(&self) -> &KMeansHyperParams {
        &self.hyperparameters
    }
}

fn compute_centroids(
    // The number of clusters could be inferred from `centroids_hashmap`,
    // but it is indeed possible for a cluster to become empty during the
    // multiple rounds of assignment-update optimisations
    // This would lead to an underestimate of the number of clusters
    // and several errors down the line due to shape mismatches
    n_clusters: usize,
    // (n_observations, n_features)
    observations: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    // (n_observations,)
    cluster_memberships: &ArrayBase<impl Data<Elem = usize>, Ix1>,
) -> Array2<f64> {
    let centroids_hashmap = compute_centroids_hashmap(&observations, &cluster_memberships);
    let (_, n_features) = observations.dim();

    let mut centroids: Array2<f64> = Array2::zeros((n_clusters, n_features));
    for (centroid_index, centroid) in centroids_hashmap.into_iter() {
        centroids
            .slice_mut(s![centroid_index, ..])
            .assign(&centroid.current_mean);
    }
    centroids
}

/// Iterate over our observations and capture in a HashMap the new centroids.
/// The HashMap is a (cluster_index => new centroid) mapping.
fn compute_centroids_hashmap(
    // (n_observations, n_features)
    observations: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    // (n_observations,)
    cluster_memberships: &ArrayBase<impl Data<Elem = usize>, Ix1>,
) -> HashMap<usize, IncrementalMean> {
    let mut new_centroids: HashMap<usize, IncrementalMean> = HashMap::new();
    Zip::from(observations.genrows())
        .and(cluster_memberships)
        .apply(|observation, cluster_membership| {
            if let Some(incremental_mean) = new_centroids.get_mut(cluster_membership) {
                incremental_mean.update(&observation);
            } else {
                new_centroids.insert(
                    *cluster_membership,
                    IncrementalMean::new(observation.to_owned()),
                );
            }
        });
    new_centroids
}

struct IncrementalMean {
    pub current_mean: Array1<f64>,
    pub n_observations: usize,
}

impl IncrementalMean {
    fn new(first_observation: Array1<f64>) -> Self {
        Self {
            current_mean: first_observation,
            n_observations: 1,
        }
    }

    fn update(&mut self, new_observation: &ArrayBase<impl Data<Elem = f64>, Ix1>) {
        self.n_observations += 1;
        let shift =
            (new_observation - &self.current_mean).mapv_into(|x| x / self.n_observations as f64);
        self.current_mean += &shift;
    }
}

fn update_cluster_memberships(
    centroids: &ArrayBase<impl Data<Elem = f64> + Sync, Ix2>,
    observations: &ArrayBase<impl Data<Elem = f64> + Sync, Ix2>,
    cluster_memberships: &mut ArrayBase<impl DataMut<Elem = usize>, Ix1>,
) {
    Zip::from(observations.axis_iter(Axis(0)))
        .and(cluster_memberships)
        .par_apply(|observation, cluster_membership| {
            *cluster_membership = closest_centroid(&centroids, &observation)
        });
}

fn compute_cluster_memberships(
    centroids: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    observations: &ArrayBase<impl Data<Elem = f64>, Ix2>,
) -> Array1<usize> {
    observations.map_axis(Axis(1), |observation| {
        closest_centroid(&centroids, &observation)
    })
}

fn closest_centroid(
    centroids: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    observation: &ArrayBase<impl Data<Elem = f64>, Ix1>,
) -> usize {
    let mut iterator = centroids.genrows().into_iter().peekable();

    let first_centroid = iterator
        .peek()
        .expect("There has to be at least one centroid");
    let (mut closest_index, mut minimum_distance) = (
        0,
        first_centroid
            .sq_l2_dist(&observation)
            .expect("Failed to compute distance"),
    );

    for (centroid_index, centroid) in iterator.enumerate() {
        let distance = centroid
            .sq_l2_dist(&observation)
            .expect("Failed to compute distance");
        if distance < minimum_distance {
            closest_index = centroid_index;
            minimum_distance = distance;
        }
    }
    closest_index
}

fn get_random_centroids<S>(
    n_clusters: usize,
    observations: &ArrayBase<S, Ix2>,
    rng: &mut impl Rng,
) -> Array2<f64>
where
    S: Data<Elem = f64>,
{
    let (n_samples, _) = observations.dim();
    let indices = rand::seq::index::sample(rng, n_samples, n_clusters).into_vec();
    observations.select(Axis(0), &indices)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{s, stack, Array, Array1, Array2, ArrayBase, Axis, Data, Ix1, Ix2};
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    /// As we highlighted several times, K-means is an iterative algorithm.
    /// We will perform the assignment and update steps until we are satisfied
    /// (according to a reasonable convergence criteria).
    ///
    /// If you go back to our `compute_cluster_memberships` function, the culmination of
    /// the assignment koan, you can see that it expects to receive centroids as a 2-dimensional
    /// array.
    ///
    /// Let's wrap our `compute_centroids_hashmap` to return a 2-dimensional array,
    /// where the i-th row corresponds to the i-th cluster.
    pub fn compute_centroids(
        n_centroids: usize,
        // (n_observations, n_features)
        observations: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        // (n_observations,)
        cluster_memberships: &ArrayBase<impl Data<Elem = usize>, Ix1>,
    ) -> Array2<f64> {
        let centroids_hashmap = compute_centroids_hashmap(&observations, &cluster_memberships);

        // Go back to "cluster generation / dataset" if you are looking for inspiration!
        let (_, n_features) = observations.dim();

        let mut centroids: Array2<f64> = Array2::zeros((n_centroids, n_features));
        for (centroid_index, centroid) in centroids_hashmap.into_iter() {
            centroids.slice_mut(s![centroid_index, ..]).assign(&centroid.current_mean);
        }
        centroids
    }

    #[test]
    fn centroids_array2() {
        let cluster_size = 100;
        let n_features = 4;

        // Let's setup a synthetic set of observations, composed of two clusters with known means
        let cluster_1: Array2<f64> =
            Array::random((cluster_size, n_features), Uniform::new(-100., 100.));
        let memberships_1 = Array1::zeros(cluster_size);
        let expected_centroid_1 = cluster_1.mean_axis(Axis(0)).unwrap();

        let cluster_2: Array2<f64> =
            Array::random((cluster_size, n_features), Uniform::new(-100., 100.));
        let memberships_2 = Array1::ones(cluster_size);
        let expected_centroid_2 = cluster_2.mean_axis(Axis(0)).unwrap();

        // `stack` combines arrays along a given axis: https://docs.rs/ndarray/0.13.0/ndarray/fn.stack.html
        let observations = stack(Axis(0), &[cluster_1.view(), cluster_2.view()]).unwrap();
        let memberships = stack(Axis(0), &[memberships_1.view(), memberships_2.view()]).unwrap();

        // Does it work?
        let centroids = compute_centroids(2, &observations, &memberships);
        assert_abs_diff_eq!(
            centroids.index_axis(Axis(0), 0),
            expected_centroid_1,
            epsilon = 1e-5
        );
        assert_abs_diff_eq!(
            centroids.index_axis(Axis(0), 1),
            expected_centroid_2,
            epsilon = 1e-5
        );

        assert_eq!(centroids.len_of(Axis(0)), 2);
    }
}
