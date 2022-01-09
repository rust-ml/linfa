use ndarray::{Array2, ArrayBase, Data, Ix1, Ix2};
use ndarray_rand::rand::Rng;
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;
use num_traits::float::FloatConst;

/// Generates a three dimension swiss roll, centered at the origin with height `height` and
/// outwards speed `speed`
pub fn generate_swissroll(
    height: f64,
    speed: f64,
    n_points: usize,
    rng: &mut impl Rng,
) -> Array2<f64> {
    let mut roll: Array2<f64> = Array2::zeros((n_points, 3));

    for i in 0..n_points {
        let z = rng.gen_range(0.0..height);
        let phi: f64 = rng.gen_range(0.0..10.0);
        //let offset: f64 = rng.gen_range(-0.5..0.5);
        let offset = 0.0;

        let x = speed * phi * phi.cos() + offset;
        let y = speed * phi * phi.sin() + offset;

        roll[(i, 0)] = x;
        roll[(i, 1)] = y;
        roll[(i, 2)] = z;
    }
    roll
}

pub fn generate_convoluted_rings(
    rings: &[(f64, f64)],
    n_points: usize,
    rng: &mut impl Rng,
) -> Array2<f64> {
    let n_points = (n_points as f32 / rings.len() as f32).ceil() as usize;
    let mut array = Array2::zeros((n_points * rings.len(), 3));

    for (n, (start, end)) in rings.iter().enumerate() {
        // inner circle
        for i in 0..n_points {
            let r: f64 = rng.gen_range(*start..*end);
            let phi: f64 = rng.gen_range(0.0..(f64::PI() * 2.0));
            let theta: f64 = rng.gen_range(0.0..(f64::PI() * 2.0));

            let x = theta.sin() * phi.cos() * r;
            let y = theta.sin() * phi.sin() * r;
            let z = theta.cos() * r;

            array[(n * n_points + i, 0)] = x;
            array[(n * n_points + i, 1)] = y;
            array[(n * n_points + i, 2)] = z;
        }
    }

    array
}

pub fn generate_convoluted_rings2d(
    rings: &[(f64, f64)],
    n_points: usize,
    rng: &mut impl Rng,
) -> Array2<f64> {
    let n_points = (n_points as f32 / rings.len() as f32).ceil() as usize;
    let mut array = Array2::zeros((n_points * rings.len(), 2));

    for (n, (start, end)) in rings.iter().enumerate() {
        // inner circle
        for i in 0..n_points {
            let r: f64 = rng.gen_range(*start..*end);
            let phi: f64 = rng.gen_range(0.0..(f64::PI() * 2.0));

            let x = phi.cos() * r;
            let y = phi.sin() * r;

            array[(n * n_points + i, 0)] = x;
            array[(n * n_points + i, 1)] = y;
        }
    }

    array
}
/// Given an input matrix `blob_centroids`, with shape `(n_blobs, n_features)`,
/// generate `blob_size` data points (a "blob") around each of the blob centroids.
///
/// More specifically, each blob is formed by `blob_size` points sampled from a normal
/// distribution centered in the blob centroid with unit variance.
///
/// `generate_blobs` can be used to quickly assemble a synthetic dataset to test or
/// benchmark various clustering algorithms on a best-case scenario input.
pub fn generate_blobs(
    blob_size: usize,
    blob_centroids: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    rng: &mut impl Rng,
) -> Array2<f64> {
    let (n_centroids, n_features) = blob_centroids.dim();
    let mut blobs: Array2<f64> = Array2::zeros((n_centroids * blob_size, n_features));

    for (blob_index, blob_centroid) in blob_centroids.rows().into_iter().enumerate() {
        let blob = generate_blob(blob_size, &blob_centroid, rng);

        let indexes = s![blob_index * blob_size..(blob_index + 1) * blob_size, ..];
        blobs.slice_mut(indexes).assign(&blob);
    }
    blobs
}
/// Generate `blob_size` data points (a "blob") around `blob_centroid`.
///
/// More specifically, the blob is formed by `blob_size` points sampled from a normal
/// distribution centered in `blob_centroid` with unit variance.
///
/// `generate_blob` can be used to quickly assemble a synthetic stereotypical cluster.
pub fn generate_blob(
    blob_size: usize,
    blob_centroid: &ArrayBase<impl Data<Elem = f64>, Ix1>,
    rng: &mut impl Rng,
) -> Array2<f64> {
    let shape = (blob_size, blob_centroid.len());
    let origin_blob: Array2<f64> = Array2::random_using(shape, StandardNormal, rng);
    origin_blob + blob_centroid
}
