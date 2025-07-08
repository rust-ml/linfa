use linfa::prelude::*;
use linfa_linear::{Result, TweedieRegressor};
use ndarray::Axis;

fn main() -> Result<(), f64> {
    // load the Diabetes dataset
    let dataset = linfa_datasets::diabetes();

    // Here the power and alpha is set to 0
    // Setting the power to 0 makes it a Normal Regressioon
    // Setting the alpha to 0 removes any regularization
    // In total this is the regular old Linear Regression
    let lin_reg = TweedieRegressor::params().power(0.).alpha(0.);
    let model = lin_reg.fit(&dataset)?;

    // We print the learnt parameters
    //
    // intercept:  152.13349207485706
    // parameters: [-10.01009490755511, -239.81838728651834, 519.8493593356682, 324.3878222341785, -792.2097759223642, 476.75394339962384, 101.07307112047873, 177.0853514839987, 751.2889123356807, 67.61902228894756]
    println!("intercept:  {}", model.intercept);
    println!("parameters: {}", model.coef);

    // We print the Mean Absolute Error (MAE) on the training data
    //
    // Some(43.27739632065444)
    let ypred = model.predict(&dataset);
    let loss = (dataset.targets() - &ypred.insert_axis(Axis(1)))
        .mapv(|x| x.abs())
        .mean();

    println!("{loss:?}");

    Ok(())
}
