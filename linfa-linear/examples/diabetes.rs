use csv::ReaderBuilder;
use ndarray::Array2;
use ndarray_csv::Array2Reader;
use std::error::Error;
use std::fs::File;
use linfa_linear::LinearRegression;

fn read_array(path: &str) -> Result<Array2<f64>, Box<dyn Error>> {
    let file = File::open(path)?;
    let mut reader = ReaderBuilder::new().has_headers(false).from_reader(file);
    let array = reader.deserialize_array2_dynamic()?;
    Ok(array)
}

fn main() -> Result<(), Box<dyn Error>> {
    let data = read_array("../datasets/diabetes_data.csv")?;
    let target = read_array("../datasets/diabetes_target.csv")?;
    let target = target.column(0).to_owned();

    let lin_reg = LinearRegression::new().with_intercept();
    let model = lin_reg.fit(&data, &target)?;

    println!("intercept:  {}", model.get_intercept());
    println!("parameters: {}", model.get_params());

    Ok(())
}