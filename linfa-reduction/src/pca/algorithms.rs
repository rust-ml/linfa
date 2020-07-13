///! Principal Component Analysis 
///
/// Reduce dimensionality with a linear projection using Singular Value Decomposition. The data is
/// centered before applying the SVD. This uses TruncatedSvd from ndarray-linalg package.

use ndarray::{ArrayBase, Array1, Array2, Ix2, Axis, DataMut};
use ndarray_linalg::{TruncatedSvd, TruncatedOrder};

/// Pincipal Component Analysis
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
        // calculate mean of data and subtract it
        let mean = dataset.mean_axis(Axis(0)).unwrap();
        dataset -= &mean;

        // estimate Singular Value Decomposition
        let result = TruncatedSvd::new(dataset.to_owned(), TruncatedOrder::Largest)
            .decompose(embedding_size)
            .unwrap();

        // explained variance is the spectral distribution of the eigenvalues
        let (_, sigma, v_t) = result.values_vectors();
        let explained_variance = sigma.mapv(|x| x*x / (dataset.len() as f64 - 1.0));

        PrincipalComponentAnalysis {
            embedding: v_t,
            explained_variance,
            mean
        }
    }

    /// Given a new data points project with fitted model
    pub fn predict<S: DataMut<Elem = f64>>(
        &self, 
        dataset: &ArrayBase<S, Ix2>
    ) -> Array2<f64> {
        (dataset - &self.mean).dot(&self.embedding.t())
    }

    /// Return the amount of explained variance per element
    pub fn explained_variance(&self) -> Array1<f64> {
        self.explained_variance.clone()
    }

    /// Return the normalized amount of explained variance per element
    pub fn explained_variance_ratio(&self) -> Array1<f64> {
        &self.explained_variance / self.explained_variance.sum()
    }
}
