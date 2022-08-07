use linfa::prelude::*;
use linfa_elasticnet::{ElasticNet, Result};

fn main() -> Result<()> {
    // load Diabetes dataset (mutable to allow fast k-folding)
    let mut dataset = linfa_datasets::diabetes();

    // parameters to compare
    let ratios = vec![0.1, 0.2, 0.5, 0.7, 1.0];

    // create a model for each parameter
    let models = ratios
        .iter()
        .map(|ratio| ElasticNet::params().penalty(0.3).l1_ratio(*ratio))
        .collect::<Vec<_>>();

    // get the mean r2 validation score across all folds for each model
    let r2_values =
        dataset.cross_validate_single(5, &models, |prediction, truth| prediction.r2(&truth))?;

    for (ratio, r2) in ratios.iter().zip(r2_values.iter()) {
        println!("L1 ratio: {}, r2 score: {}", ratio, r2);
    }

    Ok(())
}
