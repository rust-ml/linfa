use linfa::metrics::ToConfusionMatrix;
use linfa::traits::{Fit, Predict};
use linfa_bayes::{GaussianNb, Result};

fn main() -> Result<()> {
    // Read in the dataset and convert targets to binary data
    let (train, valid) = linfa_datasets::winequality()
        .map_targets(|x| if *x > 6 { "good" } else { "bad" })
        .split_with_ratio(0.9);

    // Train the model
    let model = GaussianNb::params().fit(&train)?;

    // Predict the validation dataset
    let pred = model.predict(&valid);

    // Construct confusion matrix
    let cm = pred.confusion_matrix(&valid)?;

    // classes    | bad        | good
    // bad        | 130        | 12
    // good       | 7          | 10
    //
    // accuracy 0.8805031, MCC 0.45080978
    println!("{cm:?}");
    println!("accuracy {}, MCC {}", cm.accuracy(), cm.mcc());

    Ok(())
}
