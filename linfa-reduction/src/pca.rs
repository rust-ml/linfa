//! Principal Component Analysis
//!
//! Principal Component Analysis is a common technique for data and dimensionality reduction. It
//! reduces the dimensionality of the datawhile retaining most of the variance. This is
//! done by projecting the data to a lower dimensional space with SVD and eigenvalue analysis. This
//! implementation uses the `TruncatedSvd` routine in `ndarray-linalg` which employs LOBPCG.
//!
//! # Example
//!
//! ```
//! use linfa::traits::Fit;
//! use linfa_reduction::Pca;
//!
//! let dataset = linfa_datasets::iris();
//!
//! // apply PCA projection along a line which maximizes the spread of the data
//! let embedding = Pca::params(1)
//!     .fit(&dataset);
//! ```
//!
use ndarray::{Array1, Array2, ArrayBase, Axis, Data, Ix2};
use ndarray_linalg::{TruncatedOrder, TruncatedSvd};
#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};

use linfa::{
    dataset::Targets,
    traits::{Fit, Predict},
    DatasetBase, Float,
};

/// Pincipal Component Analysis parameters
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct PcaParams {
    embedding_size: usize,
}

/// Fit a PCA model given a dataset
///
/// The Principal Component Analysis takes the records of a dataset and tries to find the best
/// fit in a lower dimensional space such that the maximal variance is retained.
///
/// # Parameters
///
/// * `dataset`: A dataset with records in N dimensions
///
/// # Returns
///
/// A dataset with records in M < N dimension reduced, such that most variance is retained
impl<'a, T: Targets> Fit<'a, Array2<f64>, T> for PcaParams {
    type Object = Pca<f64>;

    fn fit(&self, dataset: &DatasetBase<Array2<f64>, T>) -> Pca<f64> {
        let mut x = dataset.records().to_owned();
        // calculate mean of data and subtract it
        let mean = x.mean_axis(Axis(0)).unwrap();
        x -= &mean;

        // estimate Singular Value Decomposition
        let result = TruncatedSvd::new(x, TruncatedOrder::Largest)
            .decompose(self.embedding_size)
            .unwrap();

        // explained variance is the spectral distribution of the eigenvalues
        let (_, sigma, v_t) = result.values_vectors();
        let explained_variance = sigma.mapv(|x| x * x / (sigma.len() as f64 - 1.0));

        Pca {
            embedding: v_t,
            explained_variance,
            mean,
        }
    }
}

/// Fitted Principal Component Analysis model
///
/// The model contains the mean and hyperplane for the projection of data.
///
/// # Example
///
/// ```
/// use linfa::traits::Fit;
/// use linfa_reduction::Pca;
///
/// let dataset = linfa_datasets::iris();
///
/// // apply PCA projection along a line which maximizes the spread of the data
/// let embedding = Pca::params(1)
///     .fit(&dataset);
/// ```
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Debug, Clone)]
pub struct Pca<F> {
    embedding: Array2<F>,
    explained_variance: Array1<F>,
    mean: Array1<F>,
}

impl Pca<f64> {
    /// Create default parameter set
    ///
    /// # Parameters
    ///
    ///  * `embedding_size`: the target dimensionality
    pub fn params(embedding_size: usize) -> PcaParams {
        PcaParams { embedding_size }
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

/// Project a matrix to lower dimensional space
///
/// Thr projection first centers and then projects the data.
impl<F: Float, D: Data<Elem = F>> Predict<ArrayBase<D, Ix2>, Array2<F>> for Pca<F> {
    fn predict(&self, x: ArrayBase<D, Ix2>) -> Array2<F> {
        (&x - &self.mean).dot(&self.embedding.t())
    }
}
