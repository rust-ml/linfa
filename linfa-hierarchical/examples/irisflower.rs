extern crate openblas_src;

use std::error::Error;
use std::fs::File;

use csv::ReaderBuilder;
use flate2::read::GzDecoder;
use ndarray::{Array2, Axis};
use ndarray_csv::Array2Reader;

use linfa::traits::Transformer;
use linfa_hierarchical::HierarchicalCluster;
use linfa_kernel::{Kernel, KernelMethod};

/// Extract a gziped CSV file and return as dataset
fn read_array(path: &str) -> Result<Array2<f64>, Box<dyn Error>> {
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

fn main() -> Result<(), Box<dyn Error>> {
    // Read in the iris-flower dataset from dataset path
    // The `.csv` data is two dimensional: Axis(0) denotes y-axis (rows), Axis(1) denotes x-axis (columns)
    let dataset = read_array("../datasets/iris.csv.gz")?;
    let (dataset, targets) = dataset.view().split_at(Axis(1), 4);

    let kernel = Kernel::params()
        .method(KernelMethod::Gaussian(1.0))
        .transform(dataset);

    let kernel = HierarchicalCluster::default()
        .num_clusters(3)
        .transform(kernel);

    for (id, target) in kernel.targets().into_iter().zip(targets.into_iter()) {
        let name = match *target as usize {
            0 => "setosa",
            1 => "versicolor",
            2 => "virginica",
            _ => unreachable!(),
        };

        print!("({} {}) ", id, name);
    }
    println!("");

    Ok(())
}
