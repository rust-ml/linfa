use linfa_k_means as linfa_impl;
use ndarray::Array;
use ndarray_rand::rand::SeedableRng;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use rand_isaac::Isaac64Rng;

#[pyfunction]
fn k_means(
    n_clusters: usize,
    // (n_observations, n_features)
    observations: Vec<Vec<f64>>,
    tolerance: f64,
    max_n_iterations: usize,
) -> Vec<Vec<f64>> {
    // Prepare input
    let shape = (observations.len(), observations[0].len());
    let flat_observations = observations
        .into_iter()
        .flat_map(|line| line)
        .collect::<Vec<_>>();

    let observations_array = Array::from_shape_vec(shape, flat_observations).unwrap();

    // TODO: maybe receive the seed as optinal argument?
    let mut rng = Isaac64Rng::seed_from_u64(42);

    // Execute K-means
    let result = linfa_impl::k_means(
        n_clusters,
        &observations_array,
        &mut rng,
        tolerance,
        max_n_iterations,
    );

    // Prepare output
    result
        .genrows()
        .into_iter()
        .map(|row| row.to_vec())
        .collect()
}

#[pymodule]
fn linfa_k_means(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(k_means))?;

    Ok(())
}
