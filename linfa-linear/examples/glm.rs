use csv::ReaderBuilder;
use flate2::read::GzDecoder;
use linfa_linear::glm::TweedieRegressor;
use ndarray::Array2;
use ndarray_csv::Array2Reader;
use std::error::Error;
use std::fs::File;

fn main() -> Result<(), Box<dyn Error>> {
    let data = read_array("../datasets/diabetes_data.csv.gz")?;
    let target = read_array("../datasets/diabetes_target.csv.gz")?;
    let target = target.column(0).to_owned();

    let lin_reg = TweedieRegressor::new().power(0.).alpha(0.);
    let model = lin_reg.fit(&data, &target)?;

    println!("intercept:  {}", model.intercept);
    println!("parameters: {}", model.coef);

    let ypred = model.predict(&data);
    let loss = (target - ypred).mapv(|x| x.abs()).mean();
    println!("{:?}", loss);

    Ok(())
}

fn read_array(path: &str) -> Result<Array2<f64>, Box<dyn Error>> {
    let file = GzDecoder::new(File::open(path)?);
    let mut reader = ReaderBuilder::new().has_headers(false).from_reader(file);
    let array = reader.deserialize_array2_dynamic()?;
    Ok(array)
}
