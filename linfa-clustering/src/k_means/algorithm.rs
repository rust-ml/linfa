use crate::k_means::helpers::IncrementalMean;
use crate::k_means::hyperparameters::{KMeansHyperParams, KMeansHyperParamsBuilder};
use linfa::{
    dataset::{Dataset, Targets},
    traits::*,
    Float,
};
use ndarray::{s, Array1, Array2, ArrayBase, Axis, Data, DataMut, Ix1, Ix2, Zip};
use ndarray_rand::rand;
use ndarray_rand::rand::Rng;
use ndarray_stats::DeviationExt;
use rand_isaac::Isaac64Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
/// K-means clustering aims to partition a set of unlabeled observations into clusters,
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
/// extern crate openblas_src;
/// use linfa::Dataset;
/// use linfa::traits::{Fit, Predict};
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
/// let observations = Dataset::from(generate_blobs(100, &expected_centroids, &mut rng));
///
/// // Let's configure and run our K-means algorithm
/// // We use the builder pattern to specify the hyperparameters
/// // `n_clusters` is the only mandatory parameter.
/// // If you don't specify the others (e.g. `tolerance` or `max_n_iterations`)
/// // default values will be used.
/// let n_clusters = expected_centroids.len_of(Axis(0));
/// let model = KMeans::params(n_clusters)
///     .tolerance(1e-2)
///     .build()
///     .fit(&observations);
///
/// // Once we found our set of centroids, we can also assign new points to the nearest cluster
/// let new_observation = Dataset::from(array![[-9., 20.5]]);
/// // Predict returns the **index** of the nearest cluster
/// let dataset = model.predict(new_observation);
/// // We can retrieve the actual centroid of the closest cluster using `.centroids()`
/// let closest_centroid = &model.centroids().index_axis(Axis(0), dataset.targets()[0]);
/// ```
///
/*///
/// // The model can be serialised (and deserialised) to disk using serde
/// // We'll use the JSON format here for simplicity
/// let filename = "k_means_model.json";
/// let writer = std::fs::File::create(filename).expect("Failed to open file.");
/// serde_json::to_writer(writer, &model).expect("Failed to serialise model.");
///
/// let reader = std::fs::File::open(filename).expect("Failed to open file.");
/// let loaded_model: KMeans<f64> = serde_json::from_reader(reader).expect("Failed to deserialise model");
///
/// assert_abs_diff_eq!(model.centroids(), loaded_model.centroids(), epsilon = 1e-10);
/// assert_eq!(model.hyperparameters(), loaded_model.hyperparameters());
/// ```
*/
pub struct KMeans<F: Float> {
    centroids: Array2<F>,
}

impl<F: Float> KMeans<F> {
    pub fn params(nclusters: usize) -> KMeansHyperParamsBuilder<F, Isaac64Rng> {
        KMeansHyperParams::new(nclusters)
    }

    pub fn params_with_rng<R: Rng + Clone>(
        nclusters: usize,
        rng: R,
    ) -> KMeansHyperParamsBuilder<F, R> {
        KMeansHyperParams::new_with_rng(nclusters, rng)
    }

    /// Return the set of centroids as a 2-dimensional matrix with shape
    /// `(n_centroids, n_features)`.
    pub fn centroids(&self) -> &Array2<F> {
        &self.centroids
    }
}

impl<'a, F: Float, R: Rng + Clone, D: Data<Elem = F>, T: Targets> Fit<'a, ArrayBase<D, Ix2>, T>
    for KMeansHyperParams<F, R>
{
    type Object = KMeans<F>;

    /// Given an input matrix `observations`, with shape `(n_observations, n_features)`,
    /// `fit` identifies `n_clusters` centroids based on the training data distribution.
    ///
    /// An instance of `KMeans` is returned.
    ///
    fn fit(&self, dataset: &Dataset<ArrayBase<D, Ix2>, T>) -> Self::Object {
        let mut rng = self.rng();
        let observations = dataset.records().view();

        let mut centroids = get_random_centroids(self.n_clusters(), &observations, &mut rng);

        let mut has_converged;
        let mut n_iterations = 0;

        let mut memberships = Array1::zeros(observations.dim().0);

        loop {
            update_cluster_memberships(&centroids, &observations, &mut memberships);
            let new_centroids = compute_centroids(self.n_clusters(), &observations, &memberships);

            let distance = centroids
                .sq_l2_dist(&new_centroids)
                .expect("Failed to compute distance");
            has_converged = distance < self.tolerance() || n_iterations > self.max_n_iterations();

            centroids = new_centroids;
            n_iterations += 1;

            if has_converged {
                break;
            }
        }

        KMeans { centroids }
    }
}

impl<F: Float, D: Data<Elem = F>> Predict<&ArrayBase<D, Ix2>, Array1<usize>> for KMeans<F> {
    /// Given an input matrix `observations`, with shape `(n_observations, n_features)`,
    /// `predict` returns, for each observation, the index of the closest cluster/centroid.
    ///
    /// You can retrieve the centroid associated to an index using the
    /// [`centroids` method](#method.centroids).
    fn predict(&self, observations: &ArrayBase<D, Ix2>) -> Array1<usize> {
        compute_cluster_memberships(&self.centroids, observations)
    }
}

impl<F: Float, D: Data<Elem = F>, T: Targets>
    Predict<Dataset<ArrayBase<D, Ix2>, T>, Dataset<ArrayBase<D, Ix2>, Array1<usize>>>
    for KMeans<F>
{
    fn predict(
        &self,
        dataset: Dataset<ArrayBase<D, Ix2>, T>,
    ) -> Dataset<ArrayBase<D, Ix2>, Array1<usize>> {
        let predicted = self.predict(dataset.records());
        dataset.with_targets(predicted)
    }
}

/// K-means is an iterative algorithm.
/// We will perform the assignment and update steps until we are satisfied
/// (according to our convergence criteria).
///
/// If you check the `compute_cluster_memberships` function,
/// you can see that it expects to receive centroids as a 2-dimensional array.
///
/// `compute_centroids` wraps our `compute_centroids_hashmap` to return a 2-dimensional array,
/// where the i-th row corresponds to the i-th cluster.
fn compute_centroids<F: Float>(
    // The number of clusters could be inferred from `centroids_hashmap`,
    // but it is indeed possible for a cluster to become empty during the
    // multiple rounds of assignment-update optimisations
    // This would lead to an underestimate of the number of clusters
    // and several errors down the line due to shape mismatches
    n_clusters: usize,
    // (n_observations, n_features)
    observations: &ArrayBase<impl Data<Elem = F>, Ix2>,
    // (n_observations,)
    cluster_memberships: &ArrayBase<impl Data<Elem = usize>, Ix1>,
) -> Array2<F> {
    let centroids_hashmap = compute_centroids_hashmap(&observations, &cluster_memberships);
    let (_, n_features) = observations.dim();

    let mut centroids: Array2<F> = Array2::zeros((n_clusters, n_features));
    for (centroid_index, centroid) in centroids_hashmap.into_iter() {
        centroids
            .slice_mut(s![centroid_index, ..])
            .assign(&centroid.current_mean);
    }
    centroids
}

/// Iterate over our observations and capture in a HashMap the new centroids.
/// The HashMap is a (cluster_index => new centroid) mapping.
fn compute_centroids_hashmap<F: Float>(
    // (n_observations, n_features)
    observations: &ArrayBase<impl Data<Elem = F>, Ix2>,
    // (n_observations,)
    cluster_memberships: &ArrayBase<impl Data<Elem = usize>, Ix1>,
) -> HashMap<usize, IncrementalMean<F>> {
    let mut new_centroids: HashMap<usize, IncrementalMean<F>> = HashMap::new();
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

/// Given a matrix of centroids with shape (n_centroids, n_features)
/// and a matrix of observations with shape (n_observations, n_features),
/// update the 1-dimensional `cluster_memberships` array such that:
///
/// membership[i] == closest_centroid(&centroids, &observations.slice(s![i, ..])
///
fn update_cluster_memberships<F: Float>(
    centroids: &ArrayBase<impl Data<Elem = F> + Sync, Ix2>,
    observations: &ArrayBase<impl Data<Elem = F> + Sync, Ix2>,
    cluster_memberships: &mut ArrayBase<impl DataMut<Elem = usize>, Ix1>,
) {
    Zip::from(observations.axis_iter(Axis(0)))
        .and(cluster_memberships)
        .par_apply(|observation, cluster_membership| {
            *cluster_membership = closest_centroid(&centroids, &observation)
        });
}

/// Given a matrix of centroids with shape (n_centroids, n_features)
/// and a matrix of observations with shape (n_observations, n_features),
/// return a 1-dimensional `membership` array such that:
///
/// membership[i] == closest_centroid(&centroids, &observations.slice(s![i, ..])
///
fn compute_cluster_memberships<F: Float>(
    // (n_centroids, n_features)
    centroids: &ArrayBase<impl Data<Elem = F>, Ix2>,
    // (n_observations, n_features)
    observations: &ArrayBase<impl Data<Elem = F>, Ix2>,
) -> Array1<usize> {
    observations.map_axis(Axis(1), |observation| {
        closest_centroid(&centroids, &observation)
    })
}

/// Given a matrix of centroids with shape (n_centroids, n_features) and an observation,
/// return the index of the closest centroid (the index of the corresponding row in `centroids`).
fn closest_centroid<F: Float>(
    // (n_centroids, n_features)
    centroids: &ArrayBase<impl Data<Elem = F>, Ix2>,
    // (n_features)
    observation: &ArrayBase<impl Data<Elem = F>, Ix1>,
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

fn get_random_centroids<F: Float, D: Data<Elem = F>>(
    n_clusters: usize,
    observations: &ArrayBase<D, Ix2>,
    rng: &mut impl Rng,
) -> Array2<F> {
    let (n_samples, _) = observations.dim();
    let indices = rand::seq::index::sample(rng, n_samples, n_clusters).into_vec();
    observations.select(Axis(0), &indices)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{array, stack, Array, Array1, Array2, Axis};
    use ndarray_rand::rand::SeedableRng;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;
    use rand_isaac::Isaac64Rng;

    #[test]
    fn compute_centroids_works() {
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

    #[test]
    // An observation is closest to itself.
    fn nothing_is_closer_than_self() {
        let n_centroids = 20;
        let n_features = 5;
        let mut rng = Isaac64Rng::seed_from_u64(42);
        let centroids: Array2<f64> = Array::random_using(
            (n_centroids, n_features),
            Uniform::new(-100., 100.),
            &mut rng,
        );

        let expected_memberships: Vec<usize> = (0..n_centroids).into_iter().collect();
        assert_eq!(
            compute_cluster_memberships(&centroids, &centroids),
            Array1::from(expected_memberships)
        );
    }

    #[test]
    fn oracle_test_for_closest_centroid() {
        let centroids = array![[0., 0.], [1., 2.], [20., 0.], [0., 20.],];
        let observations = array![[1., 0.5], [20., 2.], [20., 0.], [7., 20.],];
        let memberships = array![0, 2, 2, 3];

        assert_eq!(
            compute_cluster_memberships(&centroids, &observations),
            memberships
        );
    }
}
