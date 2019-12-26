use crate::dbscan::hyperparameters::DbscanHyperParams;
use ndarray::{Array1, ArrayBase, ArrayView, Axis, Data, Ix1, Ix2};
use ndarray_stats::DeviationExt;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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
/// ```
/// use linfa_clustering::{DbscanHyperParams, Dbscan, generate_blobs};
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
/// // Let's configure and run our DBSCAN algorithm
/// // We use the builder pattern to specify the hyperparameters
/// // `min_points` is the only mandatory parameter.
/// // If you don't specify the others (e.g. `tolerance`)
/// // default values will be used.
/// let min_points = 3;
/// let hyperparams = DbscanHyperParams::new(min_points)
///     .tolerance(1e-2)
///     .build();
/// // Let's run the algorithm!
/// let clusters = Dbscan::predict(&hyperparams, &observations);
/// // Points are `None` if noise `Some(id)` if belonging to a cluster.
/// ```
///
pub struct Dbscan;

impl Dbscan {
    pub fn predict(
        hyperparameters: &DbscanHyperParams,
        observations: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Array1<Option<usize>> {
        let mut cluster_memberships = Array1::from_elem(observations.dim().1, None);
        let mut current_cluster_id = 0;
        for (i, obs) in observations.axis_iter(Axis(1)).enumerate() {
            if cluster_memberships[i].is_some() {
                continue;
            }
            let (neighbor_count, mut search_queue) = find_neighbors(
                &obs,
                observations,
                hyperparameters.tolerance(),
                &cluster_memberships,
            );
            if neighbor_count < hyperparameters.minimum_points() {
                continue;
            }
            // Now go over the neighbours adding them to the cluster
            cluster_memberships[i] = Some(current_cluster_id);

            while !search_queue.is_empty() {
                let candidate = search_queue.remove(0);

                let (neighbor_count, mut neighbors) = find_neighbors(
                    &candidate.1,
                    observations,
                    hyperparameters.tolerance(),
                    &cluster_memberships,
                );
                if neighbor_count >= hyperparameters.minimum_points() {
                    cluster_memberships[candidate.0] = Some(current_cluster_id);
                    search_queue.append(&mut neighbors);
                }
            }
            current_cluster_id += 1;
        }
        cluster_memberships
    }
}

fn find_neighbors<'a>(
    candidate: &ArrayBase<impl Data<Elem = f64>, Ix1>,
    observations: &'a ArrayBase<impl Data<Elem = f64>, Ix2>,
    eps: f64,
    clusters: &Array1<Option<usize>>,
) -> (usize, Vec<(usize, ArrayView<'a, f64, Ix1>)>) {
    let mut res = vec![];
    let mut count = 0;
    for (i, (obs, cluster)) in observations
        .axis_iter(Axis(1))
        .zip(clusters.iter())
        .enumerate()
    {
        if candidate.l2_dist(&obs).unwrap() < eps {
            count += 1;
            if cluster.is_none() {
                res.push((i, obs));
            }
        }
    }
    (count, res)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, s, Array2};

    #[test]
    fn nested_clusters() {
        // Create a circuit of points and then a cluster in the centre
        // and ensure they are identified as two separate clusters
        let params = DbscanHyperParams::new(2).tolerance(1.0).build();

        let mut data: Array2<f64> = Array2::zeros((2, 50));
        let rising = Array1::linspace(0.0, 8.0, 10);
        data.slice_mut(s![0, 0..10]).assign(&rising);
        data.slice_mut(s![0, 10..20]).assign(&rising);
        data.slice_mut(s![1, 20..30]).assign(&rising);
        data.slice_mut(s![1, 30..40]).assign(&rising);

        data.slice_mut(s![1, 0..10]).fill(0.0);
        data.slice_mut(s![1, 10..20]).fill(8.0);
        data.slice_mut(s![0, 20..30]).fill(0.0);
        data.slice_mut(s![0, 30..40]).fill(8.0);

        data.slice_mut(s![.., 40..]).fill(5.0);

        let labels = Dbscan::predict(&params, &data);

        assert!(labels.slice(s![..40]).iter().all(|x| x == &Some(0)));
        assert!(labels.slice(s![40..]).iter().all(|x| x == &Some(1)));
    }

    #[test]
    fn non_cluster_points() {
        let params = DbscanHyperParams::new(4).build();
        let mut data: Array2<f64> = Array2::zeros((2, 5));
        data.slice_mut(s![.., 0]).assign(&arr1(&[10.0, 10.0]));

        let labels = Dbscan::predict(&params, &data);
        let expected = arr1(&[None, Some(0), Some(0), Some(0), Some(0)]);
        assert_eq!(labels, expected);
    }

    #[test]
    fn dataset_too_small() {
        let params = DbscanHyperParams::new(4).build();

        let data: Array2<f64> = Array2::zeros((2, 3));

        let labels = Dbscan::predict(&params, &data);
        assert!(labels.iter().all(|x| x.is_none()));
    }
}
