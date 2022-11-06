//! Utility functions for randomly generating datasets

use linfa::Dataset;
use linfa::DatasetBase;
use ndarray::{s, Array, Array2, ArrayBase, Data, Dim, Ix1, Ix2, OwnedRepr};
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
#[allow(clippy::type_complexity)]
/// Generates a random Linfa::Dataset (ds). The ds values are determined by the provided statistical distributions.
///
/// # Example
/// ```
/// use statrs::distribution::{DiscreteUniform, Laplace};
/// use ndarray_rand::{RandomExt, rand_distr::Distribution};
/// use linfa_datasets::generate::make_dataset;
/// let feat_distr = Laplace::new(0.5, 5. ).unwrap();
/// let target_distr = DiscreteUniform::new(0, 5).unwrap();
/// make_dataset(5, 5, feat_distr, target_distr);
/// ```
pub fn make_dataset<X, Y>(
    num_rows: usize,
    num_feats: usize,
    feat_distr: X,
    target_distr: Y,
) -> DatasetBase<
    ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
    ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>>,
>
where
    X: Distribution<f64>,
    Y: Distribution<f64>,
{
    let targets = Array::random(num_rows, target_distr);
    let features = Array::random((num_rows, num_feats), feat_distr);

    Dataset::new(features, targets)
}
