use linfa_clustering::generate_blobs;
use linfa_trees::{DecisionTree, DecisionTreeParams, SplitQuality};
use ndarray::{array, Array};
use ndarray_rand::rand::SeedableRng;
use rand_isaac::Isaac64Rng;
use std::iter::FromIterator;

// A routine K-means task: build a synthetic dataset, fit the algorithm on it
// and save both training data and predictions to disk.
fn main() {
    // Our random number generator, seeded for reproducibility
    let mut rng = Isaac64Rng::seed_from_u64(42);

    // For each our expected centroids, generate `n` data points around it (a "blob")
    let n_classes: u64 = 3;
    let expected_centroids = array![[0., 0.], [1., 4.], [-5., 0.], [4., 4.]];
    let n = 1000;
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

    let model = DecisionTree::fit(hyperparams, &train_x, &train_y, &mut rng);

    let test_x = generate_blobs(n, &expected_centroids, &mut rng);
    let test_y = Array::from_iter(
        (0..n_classes)
            .map(|x| std::iter::repeat(x).take(n).collect::<Vec<u64>>())
            .flatten(),
    );

    let pred_y = model.predict(&test_x);
    println!("test_y: {:?}", &test_y);
    println!("pred_y: {:?}", &pred_y);
}
