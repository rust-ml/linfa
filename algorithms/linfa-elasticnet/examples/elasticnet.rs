use linfa::metrics::Regression;
use linfa::traits::{Fit, Predict};
use linfa_elasticnet::{ElasticNet, Result};

fn main() -> Result<()> {
    // load Diabetes dataset
    let (train, valid) = linfa_datasets::diabetes().split_with_ratio(0.90);

    // train pure LASSO model with 0.1 penalty
    let model = ElasticNet::params()
        .penalty(0.1)
        .l1_ratio(1.0)
        .fit(&train)?;

    println!("intercept:  {}", model.intercept());
    println!("params: {}", model.parameters());

    println!("z score: {:?}", model.z_score());

    // validate
    let y_est = model.predict(valid.records());
    println!("predicted variance: {}", valid.targets().r2(&y_est));

    Ok(())
}
