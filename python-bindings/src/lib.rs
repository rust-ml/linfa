use linfa_k_means as linfa_impl;
use linfa_k_means::KMeans;
use ndarray_rand::rand::SeedableRng;
use numpy::{PyArray1, PyArray2, ToPyArray};
use pyo3::prelude::*;
use pyo3::types::{PyString, PyType};
use rand_isaac::Isaac64Rng;
use std::path::PathBuf;

#[pyclass]
struct WrappedKMeans {
    model: KMeans,
    rng: Isaac64Rng,
}

#[pymethods]
impl WrappedKMeans {
    #[new]
    fn new(obj: &PyRawObject, random_state: Option<u64>, tolerance: f64, max_n_iterations: u64) {
        let model = linfa_impl::KMeans::new(
            Some(tolerance),
            Some(max_n_iterations),
        );
        let rng = Isaac64Rng::seed_from_u64(random_state.unwrap_or(42));
        obj.init(Self {
            model,
            rng
        });
    }

    fn fit(&mut self, n_clusters: usize, observations: &PyArray2<f64>) {
        // Prepare input
        let observations_array = observations.as_array();
        self.model.fit(n_clusters, &observations_array, &mut self.rng);
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

    fn save(&self, path: &PyString) -> PyResult<()> {
        let path = path.to_string()?;
        let path_buf = PathBuf::from(path.into_owned());
        Ok(self.model.save(path_buf)?)
    }

    #[classmethod]
    fn load(cls: &PyType, path: &PyString) -> PyResult<WrappedKMeans> {
        let path = path.to_string()?;
        let path_buf = PathBuf::from(path.into_owned());
        let model = KMeans::load(path_buf).unwrap();
        let rng = Isaac64Rng::seed_from_u64(42);
        let wrapped_model = WrappedKMeans {
            model,
            rng
        };
        return Ok(wrapped_model);
    }
}

#[pymodule]
fn linfa_k_means(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<WrappedKMeans>().unwrap();

    Ok(())
}
