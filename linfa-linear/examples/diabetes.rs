use std::error::Error;
use std::fs::File;

use csv::ReaderBuilder;
use flate2::read::GzDecoder;
use linfa_linear::LinearRegression;
use ndarray::Array2;
use ndarray_csv::Array2Reader;

use linfa::{traits::Fit, Dataset};

fn read_array(path: &str) -> Result<Array2<f64>, Box<dyn Error>> {
    let file = GzDecoder::new(File::open(path)?);
    let mut reader = ReaderBuilder::new().has_headers(false).from_reader(file);
    let array = reader.deserialize_array2_dynamic()?;
    Ok(array)
}

fn main() -> Result<(), Box<dyn Error>> {
    let data = read_array("../datasets/diabetes_data.csv.gz")?;
    let target = read_array("../datasets/diabetes_target.csv.gz")?;
    let target = target.column(0);

    let dataset = Dataset::new(data, target);

    let lin_reg = LinearRegression::new();
    let model = lin_reg.fit(&dataset)?;

    println!("intercept:  {}", model.intercept());
    println!("parameters: {}", model.params());

    Ok(())
}
