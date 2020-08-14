use std::error::Error;
use std::fs::File;

use csv::ReaderBuilder;
use flate2::read::GzDecoder;
use linfa_svm::Classification;
use ndarray::{Array2, Axis};
use ndarray_csv::Array2Reader;

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
    let data = read_array("../datasets/winequality-red.csv.gz")?;
    let (data, target) = data.view().split_at(Axis(1), 11);

    /*let lin_reg = LinearRegression::new();
    let model = lin_reg.fit(&data, &target)?;

    println!("intercept:  {}", model.intercept());
    println!("parameters: {}", model.params());*/

    Ok(())
}
