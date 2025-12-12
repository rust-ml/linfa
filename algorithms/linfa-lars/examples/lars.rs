use linfa::prelude::*;
use linfa_lars::Lars;

fn main() {
    // load Diabetes dataset
    let (train, valid) = linfa_datasets::diabetes().split_with_ratio(0.90);

    let model = Lars::params()
        .fit_intercept(true)
        .verbose(true)
        .fit(&train)
        .unwrap();

    println!("hyperplane:  {}", model.hyperplane());
    println!("intercept:  {}", model.intercept());

    // validate
    let y_est = model.predict(&valid);
    println!("predicted variance: {}", valid.r2(&y_est).unwrap());
}
