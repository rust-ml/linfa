// File: examples/iris_random_forest.rs

use linfa::prelude::*;
use linfa_datasets::iris;
use linfa_trees::{DecisionTree, RandomForestParams};
use rand::thread_rng;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Load & split Iris
    let (train, valid) = iris()
        .shuffle(&mut thread_rng())
        .split_with_ratio(0.8);

    // 2. Single‐tree baseline
    let dt_model = DecisionTree::params()
        .max_depth(None)    // no depth limit
        .fit(&train)?;
    let dt_preds = dt_model.predict(valid.records.clone());
    let dt_cm = dt_preds.confusion_matrix(&valid)?;
    println!("Single‐tree accuracy: {:.2}", dt_cm.accuracy());

    // 3. Random Forest
    let rf_model = RandomForestParams::new(50)
        .max_depth(Some(5))
        .feature_subsample(0.7)
        .fit(&train)?;
    let rf_preds = rf_model.predict(valid.records.clone());
    let rf_cm = rf_preds.confusion_matrix(&valid)?;
    println!("Random‐forest accuracy: {:.2}", rf_cm.accuracy());

    // 4. Return Ok
    Ok(())
}
