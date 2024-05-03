mod random_forest;
mod adaboost;
pub use random_forest::*;
pub use adaboost::*;

pub use linfa::error::Result;

use pyo3::prelude::*;
use linfa_trees::DecisionTreeParams;
use ndarray_rand::rand::SeedableRng;
use rand::rngs::SmallRng;

use linfa::prelude::*;

#[pyfunction]
fn train_and_predict() -> PyResult<()> {
    let mut rng = SmallRng::seed_from_u64(42);

    let (train, test) = linfa_datasets::iris()
        .shuffle(&mut rng)
        .split_with_ratio(0.8);

    println!("Training model with Adaboost ...");
    let ada_model = Adaboost::<f64,usize>::params().n_estimators(10)
        .d_tree_params(DecisionTreeParams::new().max_depth(Some(2)).min_weight_leaf(0.00001).min_weight_split(0.00001))
        .fit(&train).expect("Error");

    let ada_pred_y = ada_model.predict(&test);
    let cm = ada_pred_y.confusion_matrix(&test);

    println!("{:?}", cm);

    Ok(())
}

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn linfa_ensemble(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(train_and_predict, m)?)?;
    Ok(())
}
