use linfa::prelude::{Fit, Predict, ToConfusionMatrix};
use linfa_ensemble::EnsembleLearnerParams;
use linfa_trees::DecisionTree;
use ndarray_rand::rand::SeedableRng;
use rand::rngs::SmallRng;

fn ensemble_learner(ensemble_size: usize, bootstrap_proportion: f64, feature_proportion: f64) -> () {
    // Load dataset
    let mut rng = SmallRng::seed_from_u64(42);
    let (train, test) = linfa_datasets::iris()
        .shuffle(&mut rng)
        .split_with_ratio(0.8);

    // Train ensemble learner model
    let model = EnsembleLearnerParams::new_fixed_rng(DecisionTree::params(), rng)
        .ensemble_size(ensemble_size)
        .bootstrap_proportion(bootstrap_proportion)
        .feature_proportion(feature_proportion)
        .fit(&train)
        .unwrap();

    // Return highest ranking predictions
    let final_predictions_ensemble = model.predict(&test);
    println!("Final Predictions: \n{final_predictions_ensemble:?}");

    let cm = final_predictions_ensemble.confusion_matrix(&test).unwrap();

    println!("{cm:?}");
    println!("Test accuracy: {} \n with default Decision Tree params, \n Ensemble Size: {ensemble_size},\n Bootstrap Proportion: {bootstrap_proportion}\n Feature selection proportion: {feature_proportion}",
    100.0 * cm.accuracy());
}

fn main() {
    // This is an example bagging with decision tree
    println!("An example using Bagging with Decision Tree on Iris Dataset");
    ensemble_learner(100, 0.7, 1.0);
    println!("");
    // This is basically a Random Forest ensemble
    println!("An example using a Random Forest on Iris Dataset");
    ensemble_learner(100, 0.7, 0.2);
}
