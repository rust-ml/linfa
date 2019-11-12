use linfa_k_means as linfa_impl;
use linfa_k_means::KMeans;
use ndarray_rand::rand::SeedableRng;
use numpy::{PyArray1, PyArray2, ToPyArray};
use pyo3::prelude::*;
use rand_isaac::Isaac64Rng;

#[pyclass]
struct WrappedKMeans {
    model: KMeans<Isaac64Rng>,
}

#[pymethods]
impl WrappedKMeans {
    #[new]
    fn new(obj: &PyRawObject, random_state: Option<u64>, tolerance: f64, max_n_iterations: u64) {
        let rng = Isaac64Rng::seed_from_u64(random_state.unwrap_or(42));

        let model = linfa_impl::KMeans::new(
            Some(tolerance),
            Some(max_n_iterations),
            rng,
        );
        obj.init(Self {
            model
        });
    }

    fn fit(&mut self, n_clusters: usize, observations: &PyArray2<f64>) {
        // Prepare input
        let observations_array = observations.as_array();
        self.model.fit(n_clusters, &observations_array);
    }

    fn predict(&self, observations: &PyArray2<f64>) -> Py<PyArray1<usize>> {
        // Prepare input
        let observations_array = observations.as_array();
        let cluster_labels= self.model.predict(&observations_array);

        // Prepare output
        let gil = pyo3::Python::acquire_gil();
        let py_cluster_labels = cluster_labels.to_pyarray(gil.python()).to_owned();
        py_cluster_labels
    }

    fn centroids(&self) -> Option<Py<PyArray2<f64>>> {
        // Prepare output
        let gil = pyo3::Python::acquire_gil();
        let py_centroids = self.model.centroids().map(|x| x.to_pyarray(gil.python()).to_owned());
        py_centroids
    }
}

#[pymodule]
fn linfa_k_means(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<WrappedKMeans>();

    Ok(())
}
