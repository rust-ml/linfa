//! Utility functions for randomly generating datasets

use linfa::Dataset;
use ndarray::{s, Array, Array2, ArrayBase, Data, Ix1, Ix2};
use ndarray_rand::{
    rand::Rng,
    rand_distr::{Distribution, StandardNormal},
    RandomExt,
};

/// Special case of `blobs_with_distribution` with a standard normal distribution.
pub fn blobs(
    blob_size: usize,
    blob_centroids: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    rng: &mut impl Rng,
) -> Array2<f64> {
    blobs_with_distribution(blob_size, blob_centroids, StandardNormal, rng)
}

/// Given an input matrix `blob_centroids`, with shape `(n_blobs, n_features)`,
/// generate `blob_size` data points (a "blob") around each of the blob centroids.
///
/// More specifically, each blob is formed by `blob_size` points sampled from a distribution
/// centered in the blob centroid.
///
/// `blobs` can be used to quickly assemble a synthetic dataset to test or
/// benchmark various clustering algorithms on a best-case scenario input.
pub fn blobs_with_distribution(
    blob_size: usize,
    blob_centroids: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    distribution: impl Distribution<f64> + Clone,
    rng: &mut impl Rng,
) -> Array2<f64> {
    let (n_centroids, n_features) = blob_centroids.dim();
    let mut blobs: Array2<f64> = Array2::zeros((n_centroids * blob_size, n_features));

    for (blob_index, blob_centroid) in blob_centroids.rows().into_iter().enumerate() {
        let blob = make_blob(blob_size, &blob_centroid, distribution.clone(), rng);

        let indexes = s![blob_index * blob_size..(blob_index + 1) * blob_size, ..];
        blobs.slice_mut(indexes).assign(&blob);
    }
    blobs
}

/// Generate `blob_size` data points (a "blob") around `blob_centroid` using the given distribution.
///
/// `blob` can be used to quickly assemble a synthetic stereotypical cluster.
fn make_blob(
    blob_size: usize,
    blob_centroid: &ArrayBase<impl Data<Elem = f64>, Ix1>,
    distribution: impl Distribution<f64>,
    rng: &mut impl Rng,
) -> Array2<f64> {
    let shape = (blob_size, blob_centroid.len());
    let origin_blob: Array2<f64> = Array::random_using(shape, distribution, rng);
    origin_blob + blob_centroid
}

/// Generates a random Linfa::Dataset (ds). The ds values are determined by the provided statistical distributions.
///
/// Arguments order:
///     1 - num_rows
///     2 - num_feats
///     3 - num_targets
///     4 - feat_distr
///     5 - target_distr
///
/// # Example
/// ```
/// use statrs::distribution::{DiscreteUniform, Laplace};
/// use ndarray_rand::{RandomExt, rand_distr::Distribution};
/// let feat_distr = Laplace::new(0.5, 5. ).unwrap();
/// let target_distr = DiscreteUniform::new(0, 5).unwrap();
/// make_dataset(5, 5, 2, feat_distr, target_distr);
/// ```
fn make_dataset<X, Y>(
    num_rows: usize,
    num_feats: usize,
    num_targets: usize,
    featrandtr: X,
    target_distr: Y,
) -> Dataset<f64, f64>
where
    X: Distribution<f64>,
    Y: Distribution<f64>,
{
    let features = Array::random((num_rows, num_feats), feat_distr);
    let targets = Array::random((num_rows, num_targets), target_distr);

    Dataset::new(features, targets)
}
