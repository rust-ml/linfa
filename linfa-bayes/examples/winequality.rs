#[macro_use]
extern crate ndarray;

use std::error::Error;
use std::fs::File;

use csv::ReaderBuilder;
use flate2::read::GzDecoder;
use ndarray::{Array1, Array2, Axis};
use ndarray_csv::Array2Reader;

use linfa::dataset::Dataset;
use linfa::metrics::ToConfusionMatrix;
use linfa::traits::{Fit, Predict};
use linfa_bayes::GaussianNbParams;

fn main() -> Result<(), Box<dyn Error>> {
    // Read in the wine-quality dataset from dataset path
    // The `.csv` data is two dimensional: Axis(0) denotes y-axis (rows), Axis(1) denotes x-axis (columns)
    let dataset = read_array("../datasets/winequality-red.csv.gz")?;

    // Training data goes along the first axis (rows)
    // The first 11 columns are features used in training and the last columns are targets
    let (data, targets) = dataset.view().split_at(Axis(1), 11);
    // group targets below 6.5 as bad wine and above as good wine
    fn tag_classes(x: &f64) -> usize {
        if *x > 6.5 {
            1
        } else {
            0
        }
    };
    let targets: Array1<_> = targets.into_iter().map(tag_classes).collect();

    // Take 90% of the dataset as training data, and the remaining 10% as validation data
    // `floor` is used here to explicitely down-round the number
    let npoints = dataset.len_of(Axis(0));

    let ntrain = (npoints as f32 * 0.9).floor() as usize;
    let (train_data, train_targets) = (data.slice(s!(0..ntrain, ..)), targets.slice(s!(0..ntrain)));
    let (valid_data, valid_targets) = (data.slice(s!(ntrain.., ..)), targets.slice(s!(ntrain..)));

    let train = Dataset::new(train_data, train_targets);

    // Train the model
    let model = GaussianNbParams::params().fit(&train)?;

    let pred = model.predict(valid_data)?;

    // Construct confusion matrix
    let cm = pred.confusion_matrix(valid_targets);

    // classes    | 0          | 1
    // 0          | 131        | 7
    // 1          | 12         | 10
    //
    // accuracy 0.88125, MCC 0.45128104
    println!("{:?}", cm);
    println!("accuracy {}, MCC {}", cm.accuracy(), cm.mcc());

    Ok(())
}

// Extract a gziped CSV file and return as dataset
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
