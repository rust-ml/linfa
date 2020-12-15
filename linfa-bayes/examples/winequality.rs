use std::error::Error;

use linfa::metrics::ToConfusionMatrix;
use linfa::traits::{Fit, Predict};
use linfa_bayes::GaussianNbParams;

fn main() -> Result<(), Box<dyn Error>> {
    // Wine with rating greater than 6 is considered good
    fn tag_classes(x: &usize) -> usize {
        if *x > 6 {
            1
        } else {
            0
        }
    };

    // Read in the dataset and convert continuous target into categorical
    let data = linfa_datasets::winequality().map_targets(tag_classes);

    let (train, valid) = data.split_with_ratio_view(0.9);

    // Train the model
    let model = GaussianNbParams::params().fit(&train)?;

    let pred = model.predict(valid.records);

    // Construct confusion matrix
    let cm = pred.confusion_matrix(&valid);

    // classes    | 1          | 0
    // 1          | 10         | 12
    // 0          | 7          | 130
    //
    // accuracy 0.8805031, MCC 0.45080978
    println!("{:?}", cm);
    println!("accuracy {}, MCC {}", cm.accuracy(), cm.mcc());

    Ok(())
}
