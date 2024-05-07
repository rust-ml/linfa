#[cfg(test)]
mod tests {
    use linfa_ensemble::GBRegressor;
    use ndarray::{Array1, Array2}; // For floating-point assertions

    fn load_iris_data() -> (Array2<f64>, Array1<f64>) {
        // Load the dataset
        let dataset = linfa_datasets::iris();

        // Extract features; assuming all rows and all but the last column if last is target
        let features = dataset.records().clone();

        // Assuming the target are the labels, we need to convert them or use them as is depending on the use case.
        // If you need to predict a feature, then split differently as shown in previous messages.
        // Here we just clone the labels for the demonstration.
        let targets = dataset.targets().mapv(|x| x as f64);

        (features, targets)
    }

    #[test]
    fn test_gradient_boost_with_iris() {
        let (features, targets) = load_iris_data();

        let mut gb_regressor = GBRegressor::new(50, 0.1, 10, 3);
        gb_regressor.fit(&features, &targets);
        let predictions = gb_regressor.predict(&features);

        // Define a tolerance level
        let tolerance = 0.01; // Tolerance level for correct classification
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
fn main() {}
