#[macro_use]
extern crate ndarray;

use std::error::Error;
use std::fs::File;

use csv::ReaderBuilder;
use flate2::read::GzDecoder;
use ndarray::{Array2, Axis};
use ndarray_csv::Array2Reader;

use linfa::metrics::IntoConfusionMatrix;
use linfa_kernel::Kernel;
use linfa_svm::{SVClassify, SolverParams};

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

    // Training data goes along the first axis (rows)
    let npoints = dataset.len_of(Axis(0));
    // Take 90% of the dataset as training data, and the remaining 10% as validation data
    // `floor` is used here to explicitely down-round the number
    let ntrain = (npoints as f32 * 0.9).floor() as usize;

    // The first 11 columns are features used in training and the last columns are targets
    let (data, targets) = dataset.view().split_at(Axis(1), 11);
    // group targets below 6.5 as bad wine and above as good wine
    let targets = targets.into_iter().map(|x| *x > 6.5f64).collect::<Vec<_>>();
    // split into training and validation data
    let (train_data, train_targets) = (data.slice(s!(0..ntrain, ..)), &targets[0..ntrain]);
    let (valid_data, valid_targets) = (data.slice(s!(ntrain.., ..)), &targets[ntrain..]);

    // Transform data with gaussian kernel fuction
    // this is also known as RBF kernel with (eps = 8.0)
    let train_data = train_data.to_owned();
    let kernel = Kernel::gaussian(&train_data, 8.0);

    println!(
        "Fit SVM classifier with #{} training points",
        train_data.len_of(Axis(0))
    );

    // We will stop after convergence of `1e-3` and not shrink our variable set
    let params = SolverParams {
        eps: 1e-3,
        shrinking: false,
    };

    // Fit a support vector machine classifier with C values of `7` for negative samples and `0.6`
    // for positive, because our dataset is unbalanced
    let model = SVClassify::fit_c(&params, &kernel, train_targets, 7.0, 0.6);

    // print model
    println!("{}", model);

    // A positive prediction indicates a good wine, a negative, a bad one
    fn tag_classes(x: bool) -> String {
        if x {
            "good".into()
        } else {
            "bad".into()
        }
    };

    // Map targets from boolean to readable strings
    let valid_targets: Vec<String> = valid_targets
        .into_iter()
        .cloned()
        .map(tag_classes)
        .collect();

    // Predict the validation dataset and map to readable strings
    let prediction = valid_data
        .outer_iter()
        .map(|x| model.predict(x) > 0.0)
        .map(tag_classes)
        .collect::<Vec<_>>();

    // Convert into confusion matrix
    let cm = prediction.into_confusion_matrix(&valid_targets);

    // Print the confusion matrix, this will print a table with four entries. On the diagonal are
    // the number of true-positive and true-negative predictions, off the diagonal are
    // false-positive and false-negative
    println!("{:?}", cm);

    // Calculate the accuracy and Matthew Correlation Coefficient (cross-correlation between
    // predicted and targets)
    println!("accuracy {}, MCC {}", cm.accuracy(), cm.mcc());

    Ok(())
}
