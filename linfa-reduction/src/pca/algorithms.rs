///! Principal Component Analysis
///
/// Reduce dimensionality with a linear projection using Singular Value Decomposition. The data is
/// centered before applying the SVD. This uses TruncatedSvd from ndarray-linalg package.
use ndarray::{Array1, Array2, ArrayBase, Axis, DataMut, Ix2};
use ndarray_linalg::{Lapack, Scalar, TruncatedOrder, TruncatedSvd};

use linfa::{traits::Transformer, Float};

/// Pincipal Component Analysis
pub struct PrincipalComponentAnalysis {
    embedding_size: usize,
}

impl PrincipalComponentAnalysis {
    pub fn embedding_size(size: usize) -> PrincipalComponentAnalysis {
        PrincipalComponentAnalysis {
            embedding_size: size,
        }
    }
}

impl Transformer<Array2<f64>, PcaResult<f64>> for PrincipalComponentAnalysis {
    fn transform(&self, mut x: Array2<f64>) -> PcaResult<f64> {
        // calculate mean of data and subtract it
        let mean = x.mean_axis(Axis(0)).unwrap();
        x -= &mean;

        // estimate Singular Value Decomposition
        let result = TruncatedSvd::new(x.to_owned(), TruncatedOrder::Largest)
            .decompose(self.embedding_size)
            .unwrap();

        // explained variance is the spectral distribution of the eigenvalues
        let (_, sigma, v_t) = result.values_vectors();
        let explained_variance = sigma.mapv(|x| x * x / (sigma.len() as f64 - 1.0));

        PcaResult {
            embedding: v_t,
            explained_variance,
            mean,
        }
    }
}

pub struct PcaResult<F> {
    embedding: Array2<F>,
    explained_variance: Array1<F>,
    mean: Array1<F>,
}

impl<F: Float> PcaResult<F> {
    /// Given a new data points project with fitted model
    pub fn predict<S: DataMut<Elem = F>>(&self, dataset: &ArrayBase<S, Ix2>) -> Array2<F> {
        (dataset - &self.mean).dot(&self.embedding.t())
    }

    /// Return the amount of explained variance per element
    pub fn explained_variance(&self) -> Array1<F> {
        self.explained_variance.clone()
    }

    /// Return the normalized amount of explained variance per element
    pub fn explained_variance_ratio(&self) -> Array1<F> {
        &self.explained_variance / self.explained_variance.sum()
    }
}
