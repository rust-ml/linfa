use std::error::Error;
use std::fs::File;
use std::io::Write;

use csv::ReaderBuilder;
use flate2::read::GzDecoder;
use ndarray_csv::Array2Reader;

use ndarray::{s, Array2};
use ndarray_rand::rand::SeedableRng;
use rand_isaac::Isaac64Rng;

use linfa::prelude::*;
use linfa_trees::{DecisionTree, SplitQuality};

/// Extract a gziped CSV file and return as dataset
fn read_array(path: &str) -> std::result::Result<Array2<f64>, Box<dyn Error>> {
    // unzip file
    let file = GzDecoder::new(File::open(path)?);
    // create a CSV reader with headers and `;` as delimiter
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .delimiter(b',')
        .from_reader(file);

    // extract ndarray
    let array = reader.deserialize_array2_dynamic()?;
    Ok(array)
}

fn main() {
    // Read in the iris-flower dataset from dataset path
    // The `.csv` data is two dimensional: Axis(0) denotes y-axis (rows), Axis(1) denotes x-axis (columns)
    let dataset = read_array("../datasets/iris.csv.gz").unwrap();
    let (data, targets) = (
        dataset.slice(s![.., 0..4]).to_owned(),
        dataset.column(4).to_owned(),
    );

    let dataset = Dataset::new(data.to_owned(), targets.to_owned());
    let dataset = dataset.map_targets(|x| *x as usize);

    let mut rng = Isaac64Rng::seed_from_u64(42);
    let dataset = dataset.shuffle(&mut rng);

    let (train, test) = dataset.split_with_ratio(0.8);

    println!("Training model with Gini criterion ...");
    let gini_model = DecisionTree::params()
        .split_quality(SplitQuality::Gini)
        .max_depth(Some(100))
        .min_weight_split(1.0)
        .min_weight_leaf(1.0)
        .fit(&train);

    let gini_pred_y = gini_model.predict(test.records().view());
    let cm = gini_pred_y.confusion_matrix(&test);

    println!("{:?}", cm);

    println!(
        "Test accuracy with Gini criterion: {:.2}%",
        100.0 * cm.accuracy()
    );

    println!("Training model with entropy criterion ...");
    let entropy_model = DecisionTree::params()
        .split_quality(SplitQuality::Entropy)
        .max_depth(Some(100))
        .min_weight_split(10.0)
        .min_weight_leaf(10.0)
        .fit(&train);

    let entropy_pred_y = gini_model.predict(test.records().view());
    let cm = entropy_pred_y.confusion_matrix(&test);

    println!("{:?}", cm);

    println!(
        "Test accuracy with Entropy criterion: {:.2}%",
        100.0 * cm.accuracy()
    );

    let feats = entropy_model.features();
    println!("Features trained in this tree {:?}", feats);

    let mut tikz = File::create("decision_tree_example.tex").unwrap();
    tikz.write(gini_model.export_to_tikz().to_string().as_bytes())
        .unwrap();
    println!(" => generate tree description with `latex decision_tree_example.tex`!");
}
