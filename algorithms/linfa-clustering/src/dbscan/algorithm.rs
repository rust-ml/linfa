use crate::dbscan::{DbscanParams, DbscanValidParams};
use linfa_nn::{
    distance::{Distance, L2Dist},
    CommonNearestNeighbour, NearestNeighbour, NearestNeighbourIndex,
};
use ndarray::{Array1, ArrayBase, Data, Ix2};
use std::collections::VecDeque;

use linfa::Float;
use linfa::{traits::Transformer, DatasetBase};

#[derive(Clone, Debug, PartialEq, Eq)]
/// DBSCAN (Density-based Spatial Clustering of Applications with Noise)
/// clusters together points which are close together with enough neighbors
/// labelled points which are sparsely neighbored as noise. As points may be
/// part of a cluster or noise the predict method returns
/// `Array1<Option<usize>>`
///
/// As it groups together points in dense regions the number of clusters is
/// determined by the dataset and distance tolerance not the user.
///
/// We provide an implemention of the standard O(N^2) query-based algorithm
/// of which more details can be found in the next section or
/// [here](https://en.wikipedia.org/wiki/DBSCAN).
///
/// The standard DBSCAN algorithm isn't iterative and therefore there's
/// no fit method provided only predict.
///
/// ## The algorithm
///
/// The algorithm iterates over each point in the dataset and for every point
/// not yet assigned to a cluster:
/// - Find all points within the neighborhood of size `tolerance`
/// - If the number of points in the neighborhood is below a minimum size label
/// as noise
/// - Otherwise label the point with the cluster ID and repeat with each of the
/// neighbours
///
/// ## Tutorial
///
/// Let's do a walkthrough of an example running DBSCAN on some data.
///
/// ```rust
/// use linfa::traits::*;
/// use linfa_clustering::{DbscanParams, Dbscan};
/// use linfa_datasets::generate;
/// use ndarray::{Axis, array, s};
/// use ndarray_rand::rand::SeedableRng;
/// use rand_xoshiro::Xoshiro256Plus;
/// use approx::assert_abs_diff_eq;
///
/// // Our random number generator, seeded for reproducibility
/// let seed = 42;
/// let mut rng = Xoshiro256Plus::seed_from_u64(seed);
///
/// // `expected_centroids` has shape `(n_centroids, n_features)`
/// // i.e. three points in the 2-dimensional plane
/// let expected_centroids = array![[0., 1.], [-10., 20.], [-1., 10.]];
/// // Let's generate a synthetic dataset: three blobs of observations
/// // (100 points each) centered around our `expected_centroids`
/// let observations = generate::blobs(100, &expected_centroids, &mut rng);
///
/// // Let's configure and run our DBSCAN algorithm
/// // We use the builder pattern to specify the hyperparameters
/// // `min_points` is the only mandatory parameter.
/// // If you don't specify the others (e.g. `tolerance`)
/// // default values will be used.
/// let min_points = 3;
/// let clusters = Dbscan::params(min_points)
///     .tolerance(1e-2)
///     .transform(&observations)
///     .unwrap();
/// // Points are `None` if noise `Some(id)` if belonging to a cluster.
/// ```
///
pub struct Dbscan;

impl Dbscan {
    /// Configures the hyperparameters with the minimum number of points required to form a cluster
    ///
    /// Defaults are provided if the optional parameters are not specified:
    /// * `tolerance = 1e-4`
    /// * `dist_fn = L2Dist` (Euclidean distance)
    /// * `nn_algo = KdTree`
    pub fn params<F: Float>(min_points: usize) -> DbscanParams<F, L2Dist, CommonNearestNeighbour> {
        Self::params_with(min_points, L2Dist, CommonNearestNeighbour::KdTree)
    }

    /// Configures the hyperparameters with the minimum number of points, a custom distance metric,
    /// and a custom nearest neighbour algorithm
    pub fn params_with<F: Float, D: Distance<F>, N: NearestNeighbour>(
        min_points: usize,
        dist_fn: D,
        nn_algo: N,
    ) -> DbscanParams<F, D, N> {
        DbscanParams::new(min_points, dist_fn, nn_algo)
    }
}

impl<F: Float, D: Data<Elem = F>, DF: Distance<F>, N: NearestNeighbour>
    Transformer<&ArrayBase<D, Ix2>, Array1<Option<usize>>> for DbscanValidParams<F, DF, N>
{
    fn transform(&self, observations: &ArrayBase<D, Ix2>) -> Array1<Option<usize>> {
        let mut cluster_memberships = Array1::from_elem(observations.nrows(), None);
        let mut current_cluster_id = 0;
        // Tracks whether a value is in the search queue to prevent duplicates
        let mut search_found = vec![false; observations.nrows()];
        let mut search_queue = VecDeque::with_capacity(observations.nrows());

        // Construct NN index
        let nn = match self.nn_algo.from_batch(observations, self.dist_fn.clone()) {
            Ok(nn) => nn,
            Err(linfa_nn::BuildError::ZeroDimension) => {
                return Array1::from_elem(observations.nrows(), None)
            }
            Err(e) => panic!("Unexpected nearest neighbour error: {}", e),
        };

        for i in 0..observations.nrows() {
            if cluster_memberships[i].is_some() {
                continue;
            }
            let (neighbor_count, neighbors) =
                self.find_neighbors(&*nn, i, observations, self.tolerance, &cluster_memberships);
            if neighbor_count < self.min_points {
                continue;
            }
            neighbors.iter().for_each(|&n| search_found[n] = true);
            search_queue.extend(neighbors.into_iter());

            // Now go over the neighbours adding them to the cluster
            cluster_memberships[i] = Some(current_cluster_id);

            while let Some(candidate_idx) = search_queue.pop_front() {
                search_found[candidate_idx] = false;

                let (neighbor_count, neighbors) = self.find_neighbors(
                    &*nn,
                    candidate_idx,
                    observations,
                    self.tolerance,
                    &cluster_memberships,
                );
                // Make the candidate a part of the cluster even if it's not a core point
                cluster_memberships[candidate_idx] = Some(current_cluster_id);
                if neighbor_count >= self.min_points {
                    for n in neighbors.into_iter() {
                        if !search_found[n] {
                            search_queue.push_back(n);
                            search_found[n] = true;
                        }
                    }
                }
            }
            current_cluster_id += 1;
        }
        cluster_memberships
    }
}

impl<F: Float, D: Data<Elem = F>, DF: Distance<F>, N: NearestNeighbour, T>
    Transformer<
        DatasetBase<ArrayBase<D, Ix2>, T>,
        DatasetBase<ArrayBase<D, Ix2>, Array1<Option<usize>>>,
    > for DbscanValidParams<F, DF, N>
{
    fn transform(
        &self,
        dataset: DatasetBase<ArrayBase<D, Ix2>, T>,
    ) -> DatasetBase<ArrayBase<D, Ix2>, Array1<Option<usize>>> {
        let predicted = self.transform(dataset.records());
        dataset.with_targets(predicted)
    }
}

impl<F: Float, D: Distance<F>, N: NearestNeighbour> DbscanValidParams<F, D, N> {
    fn find_neighbors(
        &self,
        nn: &dyn NearestNeighbourIndex<F>,
        idx: usize,
        observations: &ArrayBase<impl Data<Elem = F>, Ix2>,
        eps: F,
        clusters: &Array1<Option<usize>>,
    ) -> (usize, Vec<usize>) {
        let candidate = observations.row(idx);
        let mut res = Vec::with_capacity(self.min_points);
        let mut count = 0;

        // Unwrap here is fine because we don't expect any dimension mismatch when calling
        // within_range with points from the observations
        for (_, i) in nn.within_range(candidate.view(), eps).unwrap().into_iter() {
            count += 1;
            if clusters[i].is_none() && i != idx {
                res.push(i);
            }
        }
        (count, res)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use linfa::ParamGuard;
    use linfa_nn::{distance::L1Dist, BallTree};
    use ndarray::{arr1, arr2, s, Array2};

    #[test]
    fn nested_clusters() {
        // Create a circuit of points and then a cluster in the centre
        // and ensure they are identified as two separate clusters
        let mut data: Array2<f64> = Array2::zeros((50, 2));
        let rising = Array1::linspace(0.0, 8.0, 10);
        data.column_mut(0).slice_mut(s![0..10]).assign(&rising);
        data.column_mut(0).slice_mut(s![10..20]).assign(&rising);
        data.column_mut(1).slice_mut(s![20..30]).assign(&rising);
        data.column_mut(1).slice_mut(s![30..40]).assign(&rising);

        data.column_mut(1).slice_mut(s![0..10]).fill(0.0);
        data.column_mut(1).slice_mut(s![10..20]).fill(8.0);
        data.column_mut(0).slice_mut(s![20..30]).fill(0.0);
        data.column_mut(0).slice_mut(s![30..40]).fill(8.0);

        data.column_mut(0).slice_mut(s![40..]).fill(5.0);
        data.column_mut(1).slice_mut(s![40..]).fill(5.0);

        let labels = Dbscan::params(2)
            .tolerance(1.0)
            .check()
            .unwrap()
            .transform(&data);

        assert!(labels.slice(s![..40]).iter().all(|x| x == &Some(0)));
        assert!(labels.slice(s![40..]).iter().all(|x| x == &Some(1)));
    }

    #[test]
    fn non_cluster_points() {
        let mut data: Array2<f64> = Array2::zeros((5, 2));
        data.row_mut(0).assign(&arr1(&[10.0, 10.0]));

        let labels = Dbscan::params(4).check().unwrap().transform(&data);

        let expected = arr1(&[None, Some(0), Some(0), Some(0), Some(0)]);
        assert_eq!(labels, expected);
    }

    #[test]
    fn border_points() {
        let data: Array2<f64> = arr2(&[
            // Outlier
            [0.0, 2.0],
            // Core point
            [0.0, 0.0],
            // Border points
            [0.0, 1.0],
            [0.0, -1.0],
            [-1.0, 0.0],
            [1.0, 0.0],
        ]);

        // Run the dbscan with tolerance of 1.1, 5 min points for density
        let labels = Dbscan::params(5)
            .tolerance(1.1)
            .check()
            .unwrap()
            .transform(&data);

        assert_eq!(labels[0], None);
        for id in labels.slice(s![1..]).iter() {
            assert_eq!(id, &Some(0));
        }
    }

    #[test]
    fn l1_dist() {
        let data: Array2<f64> = arr2(&[
            // Outlier
            [0.0, 6.0],
            // Core point
            [0.0, 0.0],
            // Border points
            [2.0, 3.0],
            [1.0, -3.0],
            [-4.0, 1.0],
            [1.0, 1.0],
        ]);

        // Run the L1-dist dbscan with tolerance of 5.01, 5 min points for density
        let labels = Dbscan::params_with(5, L1Dist, BallTree)
            .tolerance(5.01)
            .check()
            .unwrap()
            .transform(&data);

        assert_eq!(labels[0], None);
        for id in labels.slice(s![1..]).iter() {
            assert_eq!(id, &Some(0));
        }
    }

    #[test]
    fn dataset_too_small() {
        let data: Array2<f64> = Array2::zeros((3, 2));

        let labels = Dbscan::params(4).check().unwrap().transform(&data);
        assert!(labels.iter().all(|x| x.is_none()));
    }
}
