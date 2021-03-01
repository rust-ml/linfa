use linfa::prelude::*;
use linfa_elasticnet::{ElasticNet, Result};

fn main() -> Result<()> {
    // load Diabetes dataset
    let (train, valid) = linfa_datasets::diabetes().split_with_ratio(0.90);

    // train pure LASSO model with 0.1 penalty
    let model = ElasticNet::params()
        .penalty(0.3)
        .l1_ratio(1.0)
        .fit(&train)?;

    println!("intercept:  {}", model.intercept());
    println!("params: {}", model.parameters());

    println!("z score: {:?}", model.z_score());

    // validate
    let y_est = model.predict(&valid);
    println!("predicted variance: {}", valid.r2(&y_est));

    Ok(())
}
