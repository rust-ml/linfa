#[macro_use]
extern crate ndarray;

use std::error::Error;
use std::fs::File;

use csv::ReaderBuilder;
use flate2::read::GzDecoder;
use ndarray::{Array1, Array2, Axis};
use ndarray_csv::Array2Reader;

use linfa::metrics::IntoConfusionMatrix;
use linfa_bayes::GaussianNb;
//use linfa_kernel::Kernel;
//use linfa_svm::{SVClassify, SolverParams};

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

    println!(
        "Fit SVM classifier with #{} training points",
        train_data.len_of(Axis(0))
    );

    // A positive prediction indicates a good wine, a negative, a bad one
    fn tag_classes(x: bool) -> f64 {
        if x {
            1.
        } else {
            0.
        }
    };

    // Map targets from boolean to readable strings
    let train_targets: Array1<f64> = train_targets
        .into_iter()
        .cloned()
        .map(tag_classes)
        .collect();
    let valid_targets: Array1<f64> = valid_targets
        .into_iter()
        .cloned()
        .map(tag_classes)
        .collect();

    println!("{:?} & {:?}", valid_targets.sum(), valid_targets.len());

    let mut model: GaussianNb<f64> = GaussianNb::new();

    //let fitted_model = model.fit(&train_data, &train_targets)?;
    //let pred = fitted_model.predict(&valid_data.to_owned());

    for (x, y) in train_data
        .exact_chunks((120, 11))
        .into_iter()
        .zip(train_targets.exact_chunks(120).into_iter())
    {
        model.partial_fit(&x, &y, &array![0., 1.])?;
    }
    let fitted_model = model.get_predictor()?;
    let pred = fitted_model.predict(&valid_data.to_owned());

    let pred_str: Vec<_> = pred.to_vec().iter().map(|x| x.to_string()).collect();
    let valid_targets_str: Vec<_> = valid_targets
        .to_vec()
        .iter()
        .map(|x| x.to_string())
        .collect();

    let cm = pred_str.into_confusion_matrix(&valid_targets_str);

    println!("{:?}", cm);
    println!("accuracy {}, MCC {}", cm.accuracy(), cm.mcc());

    Ok(())
}
