use ndarray::{Array2, Array1, s};
use ndarray_csv::Array2Reader;
use std::fs::File;
use rand::rngs::StdRng;
use rand::SeedableRng;
use csv::ReaderBuilder;
use linfa_ensemble::AdaBoostRegressor;
use linfa_datasets::{boston, diabetes};

pub fn test_adaboost_with_boston_housing() {
    // Load the dataset
    let dataset = boston();  // dataset now contains both features and targets

    // Parameters for AdaBoost
    let n_estimators = 50;
    let learning_rate = 1.0;
    let max_depth = 4;
    let min_samples_split = 10;
    let random_state = 42; // Random state for reproducibility

    // Create AdaBoostRegressor instance
    let mut regressor = AdaBoostRegressor::new(n_estimators, learning_rate, random_state, max_depth, min_samples_split);

    // Fit the regressor to the Boston Housing dataset
    regressor.fit(dataset.records(), dataset.targets());

    // Make predictions
    let predictions = regressor.predict(dataset.records());

    // Calculate Mean Squared Error
    let mse = (dataset.targets() - &predictions).mapv(|a| a.powi(2)).mean().unwrap_or(0.0);  // Calculate Mean Squared Error
    let rmse = mse.sqrt();  // Calculate Root Mean Squared Error
    println!("Root Mean Squared Error for Boston Housing Dataset: {}", rmse);

    // Assert to check if RMSE is below a threshold
    assert!(rmse < 25.0, "The RMSE should be lower than 25.0, but it was {}", rmse);
}

pub fn test_adaboost_with_diabetes() {
    // Load the dataset
    let dataset = diabetes();

    // Parameters for AdaBoost
    let n_estimators = 100;  
    let learning_rate = 0.5;
    let max_depth = 3;        
    let min_samples_split = 5; 
    let random_state = 42;  

    // Create AdaBoostRegressor instance
    let mut regressor = AdaBoostRegressor::new(n_estimators, learning_rate, random_state, max_depth, min_samples_split);

    // Fit the regressor to the Diabetes dataset
    regressor.fit(dataset.records(), dataset.targets());

    // Make predictions
    let predictions = regressor.predict(dataset.records());

    // Calculate Mean Squared Error
    let mse = (dataset.targets() - &predictions).mapv(|a| a.powi(2)).mean().unwrap_or(0.0);  // Calculate Mean Squared Error
    let rmse = mse.sqrt();  // Calculate Root Mean Squared Error
    println!("Root Mean Squared Error for diabetes: {}", rmse);

    // Assert to check if RMSE is below a threshold
    assert!(rmse < 200.0, "The RMSE should be lower than 200.0, but it was {}", rmse);
}


fn main(){
    test_adaboost_with_boston_housing();
    test_adaboost_with_diabetes();
}