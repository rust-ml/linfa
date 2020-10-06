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

    let train_data = train_data.to_owned();
    let valid_data = valid_data.to_owned();

    fn tag_classes(x: bool) -> f64 {
        if x {
            1.
        } else {
            0.
        }
    };

    // Map targets from boolean to float
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

    // Initialize the model
    let mut model: GaussianNb<f64> = GaussianNb::new();

    // Trian the model using the incremental learning api
    // `fit` method is also available for training using all data
    for (x, y) in train_data
        .axis_chunks_iter(Axis(0), 120)
        .zip(train_targets.axis_chunks_iter(Axis(0), 120))
    {
        model.partial_fit(&x, &y, &array![0., 1.])?;
    }

    // Get the trained predictor
    let fitted_model = model.get_predictor()?;

    // Calculation predictions on the validation set
    let prediction = fitted_model.predict(&valid_data);

    // We convert the predictions and the validation target as string for
    // compatibility with the confusion matrix api
    let prediction_str: Vec<_> = prediction.to_vec().iter().map(|x| x.to_string()).collect();
    let valid_targets_str: Vec<_> = valid_targets
        .to_vec()
        .iter()
        .map(|x| x.to_string())
        .collect();

    let cm = prediction_str.into_confusion_matrix(&valid_targets_str);

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
