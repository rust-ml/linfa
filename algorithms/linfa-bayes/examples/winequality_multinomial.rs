use linfa::metrics::ToConfusionMatrix;
use linfa::traits::{Fit, Predict};
use linfa_bayes::{MultinomialNb, Result};

fn main() -> Result<()> {
    // Read in the dataset and convert targets to binary data
    let (train, valid) = linfa_datasets::winequality()
        .map_targets(|x| if *x > 6 { "good" } else { "bad" })
        .split_with_ratio(0.9);

    // Train the model
    let model = MultinomialNb::params().fit(&train)?;

    // Predict the validation dataset
    let pred = model.predict(&valid);

    // Construct confusion matrix
    let cm = pred.confusion_matrix(&valid)?;
    // classes    | bad        | good      
    // bad        | 88         | 54        
    // good       | 10         | 7         
    
    // accuracy 0.5974843, MCC 0.02000631
    println!("{:?}", cm);
    println!("accuracy {}, MCC {}", cm.accuracy(), cm.mcc());

    Ok(())
}
