use linfa_lars::Lars;
use linfa::prelude::*;

fn main() {

    // load Diabetes dataset
    let (train, valid) = linfa_datasets::diabetes().split_with_ratio(0.90);

    let model = Lars::params()
        .fit_intercept(false)
        .verbose(2)
        .fit(&train)
        .unwrap();
        

    println!("hyperplane:  {}", model.hyperplane());
    println!("intercept:  {}", model.intercept());

    // validate
    let y_est = model.predict(&valid);
    println!("predicted variance: {}", valid.r2(&y_est).unwrap());

}