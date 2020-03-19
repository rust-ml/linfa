use linfa_clustering::generate_blobs;
use linfa_trees::{DecisionTree, DecisionTreeParams, SplitQuality};
use ndarray::{array, Array, ArrayBase, Data, Ix1};
use ndarray_rand::rand::SeedableRng;
use rand_isaac::Isaac64Rng;
use std::iter::FromIterator;

fn accuracy(
    labels: &ArrayBase<impl Data<Elem = u64> + Sync, Ix1>,
    pred: &ArrayBase<impl Data<Elem = u64> + Sync, Ix1>,
) -> f64 {
    let mut correct = 0.0;
    let mut total = 0.0;

    for (y_label, y_pred) in labels.iter().zip(pred.iter()) {
        if y_label == y_pred {
            correct += 1.0
        }
        total += 1.0;
    }

    correct / total
}

// A routine K-means task: build a synthetic dataset, fit the algorithm on it
// and save both training data and predictions to disk.
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

    println!("Generated training data");

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
    println!("Test accuracy: {:.2}%", 100.0 * accuracy(&test_y, &pred_y));
}
