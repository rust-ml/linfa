use linfa::composing::MultiClassModel;
use linfa::prelude::*;
use linfa_svm::{error::Result, Svm};

fn main() -> Result<()> {
    let (train, valid) = linfa_datasets::winequality().split_with_ratio(0.9);

    println!(
        "Fit SVM classifier with #{} training points",
        train.nsamples()
    );

    let params = Svm::<_, Pr>::params()
        //.pos_neg_weights(5000., 500.)
        .gaussian_kernel(30.0);

    let model = train
        .one_vs_all()?
        .into_iter()
        .map(|(l, x)| (l, params.fit(&x).unwrap()))
        .collect::<MultiClassModel<_, _>>();

    let pred = model.predict(&valid);

    // create a confusion matrix
    let cm = pred.confusion_matrix(&train)?;

    // Print the confusion matrix
    println!("{:?}", cm);

    // Calculate the accuracy and Matthew Correlation Coefficient (cross-correlation between
    // predicted and targets)
    println!("accuracy {}, MCC {}", cm.accuracy(), cm.mcc());

    Ok(())
}
