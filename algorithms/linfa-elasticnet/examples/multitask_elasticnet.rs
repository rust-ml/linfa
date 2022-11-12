use linfa::prelude::*;
use linfa_elasticnet::{MultiTaskElasticNet, Result};

fn main() -> Result<()> {
    // load Diabetes dataset
    let (train, valid) = linfa_datasets::linnerud().split_with_ratio(0.80);

    // train pure LASSO model with 0.1 penalty
    let model = MultiTaskElasticNet::params()
        .penalty(0.1)
        .l1_ratio(1.0)
        .fit(&train)?;

    println!("intercept:  {}", model.intercept());
    println!("params: {}", model.hyperplane());

    println!("z score: {:?}", model.z_score());

    // validate
    let y_est = model.predict(&valid);
    println!("predicted variance: {}", y_est.r2(&valid)?);

    Ok(())
}
