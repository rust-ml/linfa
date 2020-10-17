extern crate openblas_src;

use std::error::Error;
use std::fs::File;

use csv::ReaderBuilder;
use flate2::read::GzDecoder;
use ndarray::{Array1, Array2, Axis};
use ndarray_csv::Array2Reader;

use linfa::traits::*;
use linfa::metrics::ToConfusionMatrix;
use linfa::dataset::Records;
use linfa::dataset::Dataset;
use linfa_kernel::{Kernel, KernelMethod};
use linfa_svm::Svm;

/// Extract a gziped CSV file and return as dataset
fn read_array(path: &str) -> Result<Array2<f64>, Box<dyn Error>> {
    // unzip file
    let file = GzDecoder::new(File::open(path)?);
    // create a CSV reader with headers and `;` as delimiter
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .delimiter(b';')
        .from_reader(file);
    // extract ndarray
    let array = reader.deserialize_array2_dynamic()?;
    Ok(array)
}

fn main() -> Result<(), Box<dyn Error>> {
    // Read in the wine-quality dataset from dataset path
    // The `.csv` data is two dimensional: Axis(0) denotes y-axis (rows), Axis(1) denotes x-axis (columns)
    let dataset = read_array("../datasets/winequality-red.csv.gz")?;
    // The first 11 columns are features used in training and the last columns are targets
    let (data, targets) = dataset.view().split_at(Axis(1), 11);
    let targets = targets.into_iter().collect::<Array1<_>>();

    // everything above 6.5 is considered a good wine
    let dataset = Dataset::new(data, targets)
        .map_targets(|x| **x > 6.5);

    // split into training and validation dataset
    let (train, valid) = dataset.split_with_ratio(0.1);
    
    // transform with RBF kernel
    let train_kernel = Kernel::params()
        .method(KernelMethod::Gaussian(80.0))
        .transform(&train);

    println!(
        "Fit SVM classifier with #{} training points",
        train.observations()
    );

    // fit a SVM with C value 7 and 0.6 for positive and negative classes
    let model = Svm::params()
        .pos_neg_weights(7., 0.6)
        .fit(&train_kernel);

    println!("{}", model);
    // A positive prediction indicates a good wine, a negative, a bad one
    fn tag_classes(x: &bool) -> String {
        if *x {
            "good".into()
        } else {
            "bad".into()
        }
    };

    // map targets for validation dataset
    let valid = valid.map_targets(tag_classes);

    // predict and map targets
    let pred = model.predict(&valid)
        .map_targets(|x| **x > 0.0)
        .map_targets(tag_classes);

    // create a confusion matrix
    let cm = pred.confusion_matrix(&valid);

    // Print the confusion matrix, this will print a table with four entries. On the diagonal are
    // the number of true-positive and true-negative predictions, off the diagonal are
    // false-positive and false-negative
    println!("{:?}", cm);

    // Calculate the accuracy and Matthew Correlation Coefficient (cross-correlation between
    // predicted and targets)
    println!("accuracy {}, MCC {}", cm.accuracy(), cm.mcc());

    Ok(())
}
