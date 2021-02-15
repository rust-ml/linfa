use std::error::Error;

use linfa::traits::Fit;
use linfa_linear::LinearRegression;

fn main() -> Result<(), Box<dyn Error>> {
    // load Diabetes dataset
    let dataset = linfa_datasets::diabetes();

    let lin_reg = LinearRegression::new();
    let model = lin_reg.fit(&dataset)?;

    println!("intercept:  {}", model.intercept());
    println!("parameters: {}", model.params());

    Ok(())
}
