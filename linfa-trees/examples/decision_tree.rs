use linfa_clustering::generate_blobs;
use linfa_trees::{DecisionTree, DecisionTreeParams, SplitQuality};
use ndarray::{array, Array, ArrayBase};
use ndarray_rand::rand::SeedableRng;
use rand_isaac::Isaac64Rng;
use std::iter::FromIterator;

// A routine K-means task: build a synthetic dataset, fit the algorithm on it
// and save both training data and predictions to disk.
fn main() {
    // Our random number generator, seeded for reproducibility
    let mut rng = Isaac64Rng::seed_from_u64(42);

    // For each our expected centroids, generate `n` data points around it (a "blob")
    let n_classes: u64 = 2;
    let expected_centroids = array![[10., 10.], [0., 0.]];
    let n = 5;
    let train_x = generate_blobs(n, &expected_centroids, &mut rng);
    let train_y = Array::from_iter(
        (0..n_classes)
            .map(|x| std::iter::repeat(x).take(n).collect::<Vec<u64>>())
            .flatten(),
    );

    // Configure our training algorithm
    let hyperparams = DecisionTreeParams::new(n_classes)
        .split_quality(SplitQuality::Gini)
        .max_depth(Some(100))
        .min_samples_split(50)
        .min_samples_leaf(50)
        .build();

    println!("train_x: {:?}", train_x);
    println!("train_y: {:?}", train_y);

    let model = DecisionTree::fit(hyperparams, &train_x, &train_y, &mut rng);
}
