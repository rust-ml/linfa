use std::{fs::File, io::Write};

use linfa_trees::DecisionTreeParams;
use ndarray_rand::rand::SeedableRng;
use rand::rngs::SmallRng;

use linfa::prelude::*;
use linfa_ensemble::{Adaboost, Result};

fn main() -> Result<()> {
    let mut rng = SmallRng::seed_from_u64(42);

    let (train, test) = linfa_datasets::iris()
        .shuffle(&mut rng)
        .split_with_ratio(0.8);


    println!("IRIS DATA: Training model with Adaboost ...");

    let ada_model = Adaboost::<f64, usize>::params()
        .n_estimators(10)
        .d_tree_params(
            DecisionTreeParams::new()
                .max_depth(Some(2))
                .min_weight_leaf(0.00001)
                .min_weight_split(0.00001),
        )
        .fit(&train)?;

    let ada_pred_y = ada_model.predict(&test);
    let cm = ada_pred_y.confusion_matrix(&test)?;

    println!("{:?}", cm);

    println!(
        "IRIS DATA: Test accuracy with Adaboost : {:.2}%",
        100.0 * cm.accuracy()
    );

    let mut tikz = File::create("adaboost_example.tex").unwrap();
    tikz.write_all(
        ada_model
            .export_to_tikz()
            .with_legend()
            .to_string()
            .as_bytes(),
    )
    .unwrap();

    Ok(())
}
