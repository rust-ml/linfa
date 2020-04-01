use ndarray::{ArrayBase, Array1, Array2, Ix2, Axis, DataMut};
use ndarray_linalg::{TruncatedSvd, TruncatedOrder};

use crate::Reduced;

pub struct PrincipalComponentAnalysis {
    embedding: Array2<f64>,
    explained_variance: Array1<f64>,
    mean: Array1<f64>
}

impl PrincipalComponentAnalysis {
    pub fn fit<S: DataMut<Elem = f64>>(
        mut dataset: ArrayBase<S, Ix2>,
        embedding_size: usize
    ) -> Self {
        let mean = dataset.mean_axis(Axis(0)).unwrap();

        dataset -= &mean;

        let result = TruncatedSvd::new(dataset.to_owned(), TruncatedOrder::Largest)
            .decompose(embedding_size)
            .unwrap();

        let (_, sigma, v_t) = result.values_vectors();
        let explained_variance = sigma.mapv(|x| x*x / dataset.len() as f64);

        PrincipalComponentAnalysis {
            embedding: v_t,
            explained_variance,
            mean
        }
    }

    pub fn predict<S: DataMut<Elem = f64>>(
        &self, 
        dataset: &ArrayBase<S, Ix2>
    ) -> Array2<f64> {
        (dataset - &self.mean).dot(&self.embedding.t())
    }

    pub fn explained_variance(&self) -> Array1<f64> {
        self.explained_variance.clone()
    }
}

impl Reduced for PrincipalComponentAnalysis {
    fn embedding(&self) -> Array2<f64> {
        self.embedding.clone()
    }
}
