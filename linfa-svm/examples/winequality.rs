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

fn read_array(path: &str) -> Result<Array2<f64>, Box<dyn Error>> {
    let file = GzDecoder::new(File::open(path)?);
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .delimiter(b';')
        .from_reader(file);
    let array = reader.deserialize_array2_dynamic()?;
    Ok(array)
}

fn main() -> Result<(), Box<dyn Error>> {
    // read in wine-quality dataset
    let data = read_array("../datasets/winequality-red.csv.gz")?;

    // calculate number of training and validation points
    let npoints = data.len_of(Axis(0));
    let (ntrain, nvalid) = (
        (npoints as f32 * 0.9).floor() as usize,
        (npoints as f32 * 0.1).ceil() as usize,
    );

    // last column is the target dataset
    let (data, target) = data.view().split_at(Axis(1), 11);
    // group targets below 6.5 as bad wine and above as good wine
    let target = target.into_iter().map(|x| *x > 6.5f64).collect::<Vec<_>>();
    // split into training and validation data
    let (train_data, train_target) = (data.slice(s!(0..ntrain, ..)), &target[0..ntrain]);
    let (valid_data, valid_target) = (data.slice(s!(nvalid.., ..)), &target[nvalid..]);

    // transform data with gaussian kernel fuction
    let train_data = train_data.to_owned();
    let kernel = Kernel::gaussian(&train_data, 0.1);

    // fit a support vector machine classifier with C values
    println!(
        "Fit SVM classifier with #{} training points",
        train_data.len_of(Axis(0))
    );
    let params = SolverParams {
        eps: 1e-3,
        shrinking: false,
    };

    let model = SVClassify::fit_c(&params, &kernel, train_target, 1.0, 1.0);

    // print model
    println!("{}", model);

    // a prediction "true" indicates a good wine, "false" a bad one
    fn tag_classes(x: bool) -> String {
        if x {
            "good".into()
        } else {
            "bad".into()
        }
    };

    // map targets from boolean to readable strings
    let valid_target: Vec<String> = valid_target.into_iter().cloned().map(tag_classes).collect();

    // predict, map to readable string and create a confusion matrix
    let confusion = valid_data
        .outer_iter()
        .map(|x| model.predict(x) > 0.0)
        .map(tag_classes)
        .collect::<Vec<_>>()
        .into_confusion_matrix(&valid_target);

    println!("{:?}", confusion);
    println!("accuracy {}, MCC {}", confusion.accuracy(), confusion.mcc());

    Ok(())
}
