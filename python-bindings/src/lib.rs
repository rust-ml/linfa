use linfa_clustering::{KMeans, KMeansHyperParams};
use ndarray_rand::rand::SeedableRng;
use numpy::{PyArray1, PyArray2, ToPyArray};
use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::{PyString, PyType};
use rand_isaac::Isaac64Rng;
use serde::{Deserialize, Serialize};

#[pyclass]
#[derive(Serialize, Deserialize)]
struct WrappedKMeans {
    hyperparams: KMeansHyperParams,
    rng: Isaac64Rng,
    model: Option<KMeans>,
}

#[pymethods]
impl WrappedKMeans {
    #[new]
    fn new(
        obj: &PyRawObject,
        n_clusters: usize,
        random_state: Option<u64>,
        tolerance: f64,
        max_n_iterations: u64,
    ) {
        let hyperparams = KMeansHyperParams::new(n_clusters)
            .tolerance(tolerance)
            .max_n_iterations(max_n_iterations)
            .build();
        let rng = Isaac64Rng::seed_from_u64(random_state.unwrap_or(42));
        obj.init(Self {
            hyperparams,
            rng,
            // Populated when after `fit` has been called
            model: None,
        });
    }

    fn fit(&mut self, observations: &PyArray2<f64>) {
        // Prepare input
        let observations_array = observations.as_array();
        let model = KMeans::fit(self.hyperparams.clone(), &observations_array, &mut self.rng);
        self.model = Some(model);
    }

    fn predict(&self, observations: &PyArray2<f64>) -> Option<Py<PyArray1<usize>>> {
        // Prepare input
        let observations_array = observations.as_array();
        let cluster_labels = self.model.as_ref().map(|m| m.predict(&observations_array));

        // Prepare output
        let gil = pyo3::Python::acquire_gil();
        let py_cluster_labels = cluster_labels.map(|c| c.to_pyarray(gil.python()).to_owned());
        py_cluster_labels
    }

    fn centroids(&self) -> Option<Py<PyArray2<f64>>> {
        // Prepare output
        let gil = pyo3::Python::acquire_gil();
        let py_centroids = self
            .model
            .as_ref()
            .map(|m| m.centroids().to_pyarray(gil.python()).to_owned());
        py_centroids
    }

    fn save(&self, path: &PyString) -> PyResult<()> {
        let path = path.to_string()?;
        let writer = std::fs::File::create(path.into_owned())?;
        serde_json::to_writer(writer, &self)
            .map_err(|e| PyErr::new::<exceptions::Exception, _>(e.to_string()))
    }

    #[classmethod]
    fn load(_cls: &PyType, path: &PyString) -> PyResult<WrappedKMeans> {
        let path = path.to_string()?;
        let reader = std::fs::File::open(path.into_owned())?;
        serde_json::from_reader(reader)
            .map_err(|e| PyErr::new::<exceptions::Exception, _>(e.to_string()))
    }
}

#[pymodule]
fn linfa_k_means(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<WrappedKMeans>().unwrap();

    Ok(())
}
