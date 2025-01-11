use linfa::prelude::*;
use linfa_logistic::MultiLogisticRegression;

use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let (train, valid) = linfa_datasets::winequality().split_with_ratio(0.9);

    println!(
        "Fit Multinomial Logistic Regression classifier with #{} training points",
        train.nsamples()
    );

    // fit a Logistic regression model with 150 max iterations
    let model = MultiLogisticRegression::default()
        .max_iterations(50)
        .fit(&train)
        .unwrap();

    // predict and map targets
    let pred = model.predict(&valid);

    // create a confusion matrix
    let cm = pred.confusion_matrix(&valid).unwrap();

    // Print the confusion matrix, this will print a table with four entries. On the diagonal are
    // the number of true-positive and true-negative predictions, off the diagonal are
    // false-positive and false-negative
    println!("{:?}", cm);

    // Calculate the accuracy and Matthew Correlation Coefficient (cross-correlation between
    // predicted and targets)
    println!("accuracy {}, MCC {}", cm.accuracy(), cm.mcc());

    Ok(())
}
