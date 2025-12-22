use linfa::prelude::{Fit, Predict, ToConfusionMatrix};
use linfa_ensemble::AdaBoostParams;
use linfa_trees::DecisionTree;
use ndarray_rand::rand::SeedableRng;
use rand::rngs::SmallRng;

fn adaboost_with_stumps(n_estimators: usize, learning_rate: f64) {
    // Load dataset
    let mut rng = SmallRng::seed_from_u64(42);
    let (train, test) = linfa_datasets::iris()
        .shuffle(&mut rng)
        .split_with_ratio(0.8);

    // Train AdaBoost model with decision tree stumps (max_depth=1)
    // Stumps are weak learners commonly used with AdaBoost
    let model = AdaBoostParams::new_fixed_rng(DecisionTree::params().max_depth(Some(1)), rng)
        .n_estimators(n_estimators)
        .learning_rate(learning_rate)
        .fit(&train)
        .unwrap();

    // Make predictions
    let predictions = model.predict(&test);
    println!("Final Predictions: \n{predictions:?}");

    let cm = predictions.confusion_matrix(&test).unwrap();
    println!("{cm:?}");
    println!(
        "Test accuracy: {:.2}%\nwith Decision Tree stumps (max_depth=1),\nn_estimators: {n_estimators},\nlearning_rate: {learning_rate}.\n",
        100.0 * cm.accuracy()
    );
    println!("Number of models trained: {}", model.n_estimators());
}

fn adaboost_with_shallow_trees(n_estimators: usize, learning_rate: f64, max_depth: usize) {
    let mut rng = SmallRng::seed_from_u64(42);
    let (train, test) = linfa_datasets::iris()
        .shuffle(&mut rng)
        .split_with_ratio(0.8);

    // Train AdaBoost model with shallow decision trees
    let model =
        AdaBoostParams::new_fixed_rng(DecisionTree::params().max_depth(Some(max_depth)), rng)
            .n_estimators(n_estimators)
            .learning_rate(learning_rate)
            .fit(&train)
            .unwrap();

    // Make predictions
    let predictions = model.predict(&test);
    println!("Final Predictions: \n{predictions:?}");

    let cm = predictions.confusion_matrix(&test).unwrap();
    println!("{cm:?}");
    println!(
        "Test accuracy: {:.2}%\nwith Decision Trees (max_depth={max_depth}),\nn_estimators: {n_estimators},\nlearning_rate: {learning_rate}.\n",
        100.0 * cm.accuracy()
    );

    // Display model weights
    println!("Model weights (alpha values):");
    for (i, weight) in model.weights().iter().enumerate() {
        println!("  Model {}: {:.4}", i + 1, weight);
    }
    println!();
}

fn main() {
    println!("{}", "=".repeat(80));
    println!("AdaBoost Examples on Iris Dataset");
    println!("{}", "=".repeat(80));
    println!();

    // Example 1: AdaBoost with decision stumps (most common configuration)
    println!("Example 1: AdaBoost with Decision Stumps");
    println!("{}", "-".repeat(80));
    adaboost_with_stumps(50, 1.0);
    println!();

    // Example 2: AdaBoost with lower learning rate
    println!("Example 2: AdaBoost with Lower Learning Rate");
    println!("{}", "-".repeat(80));
    adaboost_with_stumps(100, 0.5);
    println!();

    // Example 3: AdaBoost with shallow trees
    println!("Example 3: AdaBoost with Shallow Decision Trees");
    println!("{}", "-".repeat(80));
    adaboost_with_shallow_trees(50, 1.0, 2);
    println!();

    // Example 4: Comparing different configurations
    println!("Example 4: Comparing Configurations");
    println!("{}", "-".repeat(80));
    let configs = vec![
        (25, 1.0, 1, "Few stumps, high learning rate"),
        (50, 1.0, 1, "Medium stumps, high learning rate"),
        (100, 0.5, 1, "Many stumps, low learning rate"),
        (50, 1.0, 2, "Shallow trees, high learning rate"),
    ];

    for (n_est, lr, depth, desc) in configs {
        let mut rng = SmallRng::seed_from_u64(42);
        let (train, test) = linfa_datasets::iris()
            .shuffle(&mut rng)
            .split_with_ratio(0.8);

        let model =
            AdaBoostParams::new_fixed_rng(DecisionTree::params().max_depth(Some(depth)), rng)
                .n_estimators(n_est)
                .learning_rate(lr)
                .fit(&train)
                .unwrap();

        let predictions = model.predict(&test);
        let cm = predictions.confusion_matrix(&test).unwrap();

        println!(
            "{desc:50} => Accuracy: {:.2}% (models trained: {})",
            100.0 * cm.accuracy(),
            model.n_estimators()
        );
    }

    println!();
    println!("{}", "=".repeat(80));
    println!("Notes:");
    println!("- AdaBoost works by training weak learners sequentially");
    println!("- Each learner focuses on samples misclassified by previous learners");
    println!("- Decision stumps (depth=1) are the most common weak learners");
    println!("- Lower learning_rate provides regularization but needs more estimators");
    println!("- Model weights (alpha) reflect each learner's contribution to prediction");
    println!("{}", "=".repeat(80));
}
