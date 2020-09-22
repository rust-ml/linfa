use linfa_clustering::generate_blobs;
use linfa_predictor::Predictor;
use linfa_trees::{DecisionTree, DecisionTreeParams, SplitQuality};
use ndarray::{array, Array, ArrayBase, Data, Ix1};
use ndarray_rand::rand::SeedableRng;
use rand_isaac::Isaac64Rng;
use std::iter::FromIterator;

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

    let gini_pred_y = gini_model.predict(&test_x).unwrap();
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

    let entropy_pred_y = entropy_model.predict(&test_x).unwrap();
    println!(
        "Test accuracy with Entropy criterion: {:.2}%",
        100.0 * accuracy(&test_y, &entropy_pred_y)
    );

    let feats = entropy_model.features();
    println!("Features trained in this tree {:?}", feats);
}
