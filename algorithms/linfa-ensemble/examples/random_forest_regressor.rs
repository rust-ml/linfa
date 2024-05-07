// use linfa_ensemble::RandomForestRegressor;
// use ndarray::{Array1, Axis};
// use rand::seq::SliceRandom;
// use rand::thread_rng;
// use linfa_ensemble::visualization;

// fn main() {
//     // Number of trees in the forest
//     let num_trees = 100;
//     // Number of features to consider for each split
//     let max_features = 4; // Set to the number of features in your dataset or adjust as needed
//     // Maximum depth of each tree
//     let max_depth = 10;
//     // Minimum number of samples required to split a node
//     let min_samples_split = 5;

//     // Load the Iris dataset
//     let iris = linfa_datasets::diabetes();
//     let iris_cloned = iris.clone();

//     // Extract features and targets
//     let features = iris_cloned.records();
//     let targets = iris.targets().mapv(|x| x as f64);

//     // Shuffle and split the data into train and test
//     let mut rng = thread_rng();
//     let mut indices: Vec<usize> = (0..features.nrows()).collect();
//     indices.shuffle(&mut rng);
//     let split_index = (features.nrows() as f64 * 0.8) as usize; // 60% train, 40% test
//     let train_indices = &indices[..split_index];
//     let test_indices = &indices[split_index..];

//     let train_features = features.select(Axis(0), train_indices);
//     let train_targets = targets.select(Axis(0), train_indices);
//     let test_features = features.select(Axis(0), test_indices);
//     let test_targets = targets.select(Axis(0), test_indices);

//     // Train random forest regressor
//     let mut forest = RandomForestRegressor::new(num_trees, max_features, max_depth, min_samples_split);
//     forest.fit(&train_features, &train_targets);

//     // Predict on test dataset
//     let predictions = forest.predict(&test_features);

//     // Evaluate performance
//     let mse = mean_squared_error(&test_targets, &predictions);
//     println!("Mean Squared Error: {}", mse);



//     println!("Generated graph");
// }

// fn mean_squared_error(actual: &Array1<f64>, predicted: &Array1<f64>) -> f64 {
//     let errors = actual - predicted;
//     let squared_errors = errors.mapv(|x| x.powi(2));
//     squared_errors.mean().unwrap()
// }

#[cfg(test)]
mod tests {
    use super::*;
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

    #[test]
    fn test_random_forest_with_diabetes() {
         let (features, targets) = load_diabetes_data();

        // Split data into training and testing sets
        let split_ratio = 0.7; // Using 70% of the data for training
        let split_index = (features.nrows() as f64 * split_ratio) as usize;
        let (train_features, test_features) = features.view().split_at(Axis(0), split_index);
        let (train_targets, test_targets) = targets.view().split_at(Axis(0), split_index);

        let mut forest = RandomForestRegressor::new(100, 10, 5, 10);
        // Convert views to owned arrays before passing to fit
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


    #[test]
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

        println!("RMSE: {:?}", rmse);
    }
}
