use linfa_trees::{DecisionTree, DecisionTreeParams, SplitQuality};
use ndarray::{array, Array, ArrayBase, Data, Ix1, Ix2, Array2, s};
use ndarray_rand::rand::Rng;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;
use rand_isaac::Isaac64Rng;
use std::iter::FromIterator;

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

    for (blob_index, blob_centroid) in blob_centroids.genrows().into_iter().enumerate() {
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
    let origin_blob: Array2<f64> = Array::random_using(shape, StandardNormal, rng);
    origin_blob + blob_centroid
}

fn accuracy(
    labels: &ArrayBase<impl Data<Elem = u64>, Ix1>,
    pred: &ArrayBase<impl Data<Elem = u64>, Ix1>,
) -> f64 {
    let true_positive: f64 = labels
        .iter()
        .zip(pred.iter())
        .filter(|(x, y)| x == y)
        .map(|_| 1.0)
        .sum();
    true_positive / labels.len() as f64
}

fn main() {
    // Our random number generator, seeded for reproducibility
    let mut rng = Isaac64Rng::seed_from_u64(42);

    // For each our expected centroids, generate `n` data points around it (a "blob")
    let n_classes: u64 = 4;
    let expected_centroids = array![[0., 0.], [1., 4.], [-5., 0.], [4., 4.]];
    let n = 100;

    println!("Generating training data");

    let train_x = generate_blobs(n, &expected_centroids, &mut rng);
    let train_y = Array::from_iter(
        (0..n_classes)
            .map(|x| std::iter::repeat(x).take(n).collect::<Vec<u64>>())
            .flatten(),
    );

    let test_x = generate_blobs(n, &expected_centroids, &mut rng);
    let test_y = Array::from_iter(
        (0..n_classes)
            .map(|x| std::iter::repeat(x).take(n).collect::<Vec<u64>>())
            .flatten(),
    );

    println!("Generated training data");

    println!("Training model with Gini criterion ...");
    let gini_hyperparams = DecisionTreeParams::new(n_classes)
        .split_quality(SplitQuality::Gini)
        .max_depth(Some(100))
        .min_samples_split(10)
        .min_samples_leaf(10)
        .build();

    let gini_model = DecisionTree::fit(gini_hyperparams, &train_x, &train_y);

    let gini_pred_y = gini_model.predict(&test_x);
    println!(
        "Test accuracy with Gini criterion: {:.2}%",
        100.0 * accuracy(&test_y, &gini_pred_y)
    );

    println!("Training model with entropy criterion ...");
    let entropy_hyperparams = DecisionTreeParams::new(n_classes)
        .split_quality(SplitQuality::Entropy)
        .max_depth(Some(100))
        .min_samples_split(10)
        .min_samples_leaf(10)
        .build();

    let entropy_model = DecisionTree::fit(entropy_hyperparams, &train_x, &train_y);

    let entropy_pred_y = entropy_model.predict(&test_x);
    println!(
        "Test accuracy with Entropy criterion: {:.2}%",
        100.0 * accuracy(&test_y, &entropy_pred_y)
    );
}
