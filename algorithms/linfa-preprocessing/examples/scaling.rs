use linfa::metrics::ToConfusionMatrix;
use linfa::traits::{Fit, Predict, Transformer};
use linfa_bayes::GaussianNbParams;
use linfa_preprocessing::linear_scaling::LinearScaler;

fn main() {
    // Read in the dataset and convert continuous target into categorical
    let (train, valid) = linfa_datasets::winequality()
        .map_targets(|x| if *x > 6 { 1 } else { 0 })
        .split_with_ratio(0.7);

    // Fit a standard scaler to the training set
    let scaler = LinearScaler::standard().fit(&train).unwrap();

    // Scale training and validation sets according to the fitted scaler
    let train = scaler.transform(train);
    let valid = scaler.transform(valid);

    // Learn a naive bayes model from the training set
    let model = GaussianNbParams::params().fit(&train).unwrap();

    // compute accuracies
    let train_acc = model
        .predict(&train)
        .confusion_matrix(&train)
        .unwrap()
        .accuracy();
    let cm = model.predict(&valid).confusion_matrix(&valid).unwrap();
    let valid_acc = cm.accuracy();
    println!(
        "Scaled model training and validation accuracies: {} - {}",
        train_acc, valid_acc
    );
    println!("{:?}", cm);
}
