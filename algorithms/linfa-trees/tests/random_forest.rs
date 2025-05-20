// linfa-trees/tests/random_forest.rs

use linfa::prelude::*;
use linfa_datasets::iris;
use linfa_trees::RandomForestParams;
use rand::rngs::StdRng;
use rand::SeedableRng;

#[test]
fn iris_random_forest_high_accuracy() {
    // reproducible split
    let mut rng = StdRng::seed_from_u64(42);
    let (train, valid) = iris().shuffle(&mut rng).split_with_ratio(0.8);

    let model = RandomForestParams::new(100)
        .max_depth(Some(10))
        .feature_subsample(0.8)
        .seed(42)
        .fit(&train)
        .expect("Training failed");

    let preds = model.predict(valid.records.clone());
    let cm = preds
        .confusion_matrix(&valid)
        .expect("Failed to compute confusion matrix");

    let accuracy = cm.accuracy();
    assert!(
        accuracy >= 0.9,
        "Expected ≥90% accuracy on Iris, got {:.2}",
        accuracy
    );
}
