mod adaboost;
mod random_forest;

// mod random_forest_regressor;
mod gradient_boost;
pub mod visualization;

pub use adaboost::*;
pub use random_forest::*;
pub use visualization::*;
// pub use random_forest_regressor::*;
pub use gradient_boost::*;

pub use linfa::error::Result;

use linfa_trees::{DecisionTree, DecisionTreeParams};
use ndarray_rand::rand::SeedableRng;
use pyo3::prelude::*;
use rand::rngs::SmallRng;

use linfa::prelude::*;

#[pyfunction]
fn train_and_predict_randomforest() -> PyResult<()> {
    let ensemble_size = 100;
    //Proportion of training data given to each model
    let bootstrap_proportion = 0.7;

    //Load dataset
    let mut rng = SmallRng::seed_from_u64(42);
    let (train, test) = linfa_datasets::iris()
        .shuffle(&mut rng)
        .split_with_ratio(0.7);

    //Train ensemble learner model
    let model = EnsembleLearnerParams::new(DecisionTree::<f64, usize>::params())
        .ensemble_size(ensemble_size)
        .bootstrap_proportion(bootstrap_proportion)
        .fit(&train)
        .unwrap();
    // println!("Done with Fit");
    //   //Return highest ranking predictions
    let final_predictions_ensemble = model.predict(&test);
    println!("Final Predictions: \n{:?}", final_predictions_ensemble);

    let cm = final_predictions_ensemble.confusion_matrix(&test).unwrap();

    println!("{:?}", cm);
    println!("Test accuracy: {} \n with default Decision Tree params, \n Ensemble Size: {},\n Bootstrap Proportion: {}",
  100.0 * cm.accuracy(), ensemble_size, bootstrap_proportion);
    Ok(())
}

#[pyfunction]
fn train_and_predict_adaboost() -> PyResult<()> {
    let mut rng = SmallRng::seed_from_u64(42);

    let (train, test) = linfa_datasets::iris()
        .shuffle(&mut rng)
        .split_with_ratio(0.8);

    println!("Training model with Adaboost ...");
    let ada_model = Adaboost::<f64, usize>::params()
        .n_estimators(10)
        .d_tree_params(
            DecisionTreeParams::new()
                .max_depth(Some(2))
                .min_weight_leaf(0.00001)
                .min_weight_split(0.00001),
        )
        .fit(&train)
        .expect("Error");

    let ada_pred_y = ada_model.predict(&test);
    let cm = ada_pred_y.confusion_matrix(&test);

    println!("{:?}", cm);

    Ok(())
}

/// A Python module implemented in Rust.
#[pymodule]
fn linfa_ensemble(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(train_and_predict_randomforest, m)?)?;
    m.add_function(wrap_pyfunction!(train_and_predict_adaboost, m)?)?;
    Ok(())
}
