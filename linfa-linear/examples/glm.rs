use csv::ReaderBuilder;
use flate2::read::GzDecoder;
use linfa_linear::TweedieRegressor;
use ndarray::Array2;
use ndarray_csv::Array2Reader;
use std::error::Error;
use std::fs::File;

fn main() -> Result<(), Box<dyn Error>> {
    let data = read_array("../datasets/diabetes_data.csv.gz")?;
    let target = read_array("../datasets/diabetes_target.csv.gz")?;
    let target = target.column(0).to_owned();

    // Here the power and alpha is set to 0
    // Setting the power to 0 makes it a Normal Regressioon
    // Setting the alpha to 0 removes any regularization
    // In total this is the regular old Linear Regression
    let lin_reg = TweedieRegressor::new().power(0.).alpha(0.);
    let model = lin_reg.fit(&data, &target)?;

    // We print the learnt parameters
    //
    // intercept:  152.13349207485706
    // parameters: [-10.01009490755511, -239.81838728651834, 519.8493593356682, 324.3878222341785, -792.2097759223642, 476.75394339962384, 101.07307112047873, 177.0853514839987, 751.2889123356807, 67.61902228894756]
    println!("intercept:  {}", model.intercept);
    println!("parameters: {}", model.coef);

    // We print the Mean Absolute Error (MAE) on the training data
    //
    // Some(43.27739632065444)
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
