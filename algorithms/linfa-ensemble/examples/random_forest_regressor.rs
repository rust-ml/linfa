use approx::assert_relative_eq;
use linfa_datasets::{iris, diabetes};
use linfa_ensemble::RandomForestRegressor;
use ndarray::{Array1, Array2, Axis}; // For floating-point assertions
use linfa_ensemble::visualization;

fn calculate_rmse(actual: &Array1<f64>, predicted: &Array1<f64>) -> f64 {
    let errors = actual - predicted;
    let mse = errors.mapv(|e| e.powi(2)).mean().unwrap();
    mse.sqrt()
}


fn load_iris_data() -> (Array2<f64>, Array1<f64>) {
    // Load the dataset
    let dataset = iris();

    // Extract features; assuming all rows and all but the last column if last is target
    let features = dataset.records().clone();

    let targets = dataset.targets().mapv(|x| x as f64);

    (features, targets)
}

fn load_diabetes_data() -> (Array2<f64>, Array1<f64>) {
    let dataset = diabetes();

    let features = dataset.records().clone();
    let targets = dataset.targets().mapv(|x| x as f64);

    (features, targets)
}

fn test_random_forest_with_diabetes() {
    let (features, targets) = load_diabetes_data();

    // Split data into training and testing sets
    let split_ratio = 0.7; // Using 70% of the data for training
    let split_index = (features.nrows() as f64 * split_ratio) as usize;
    let (train_features, test_features) = features.view().split_at(Axis(0), split_index);
    let (train_targets, test_targets) = targets.view().split_at(Axis(0), split_index);

    let mut forest = RandomForestRegressor::new(150, 10, 5, 10);
    forest.fit(&train_features.to_owned(), &train_targets.to_owned());
    let train_predictions = forest.predict(&train_features.to_owned());
    let test_predictions = forest.predict(&test_features.to_owned());

    // Evaluate the performance on the test set
    let test_rmse = calculate_rmse(&test_targets.to_owned(), &test_predictions);
    println!("Test RMSE for Diabetes Dataset: {:?}", test_rmse);

    // Assert that the RMSE is below an acceptable threshold
    assert!(test_rmse < 70.0, "The RMSE should be lower than 60.0");

    // Visualization of training and testing results
    visualization::plot_scatter(
        &train_targets.to_owned(),
        &train_predictions,
        &test_targets.to_owned(),
        &test_predictions,
        "diabetes_rf_scatter.png",
    ).unwrap();
}

fn test_random_forest_with_iris() {
    let (features, targets) = load_iris_data();

    let mut forest = RandomForestRegressor::new(100, 10, 3, 10);
    forest.fit(&features, &targets);
    let predictions = forest.predict(&features);


    // Define a tolerance level
    let tolerance = 0.1; // Tolerance level for correct classification
    let mut correct = 0;
    let mut incorrect = 0;

    // Count correct and incorrect predictions
    for (&actual, &predicted) in targets.iter().zip(predictions.iter()) {
        if (predicted - actual).abs() < tolerance {
            correct += 1;
        } else {
            incorrect += 1;
        }
    }

    println!("Correct predictions: {}", correct);
    println!("Incorrect predictions: {}", incorrect);

    let rmse = (&predictions - &targets)
        .mapv(|a| a.powi(2))
        .mean()
        .unwrap()
        .sqrt();

    println!("Test RMSE for Iris Dataset: {:?}", rmse);
}


fn main() {
    test_random_forest_with_iris();
    test_random_forest_with_diabetes();
}
