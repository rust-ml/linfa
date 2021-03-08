use linfa::metrics::ToConfusionMatrix;
use linfa::traits::{Fit, Predict};
use linfa_bayes::{GaussianNbParams, Result};

fn main() -> Result<()> {
    // Read in the dataset and convert continuous target into categorical
    let (train, valid) = linfa_datasets::winequality()
        .map_targets(|x| if *x > 6 { 1 } else { 0 })
        .split_with_ratio(0.9);

    // Train the model
    let model = GaussianNbParams::params().fit(&train.view())?;

    // Predict the validation dataset
    let pred = model.predict(&valid);

    // Construct confusion matrix
    let cm = pred.confusion_matrix(&valid)?;

    // classes    | 1          | 0
    // 1          | 10         | 12
    // 0          | 7          | 130
    //
    // accuracy 0.8805031, MCC 0.45080978
    println!("{:?}", cm);
    println!("accuracy {}, MCC {}", cm.accuracy(), cm.mcc());

    Ok(())
}
