//! Principal Component Analysis
//!
//! Principal Component Analysis is a common technique for data and dimensionality reduction. It
//! reduces the dimensionality of the data while retaining most of the variance. This is
//! done by projecting the data to a lower dimensional space with SVD and eigenvalue analysis. This
//! implementation uses the `TruncatedSvd` routine in `ndarray-linalg` which employs LOBPCG.
//!
//! # Example
//!
//! ```
//! use linfa::traits::{Fit, Predict};
//! use linfa_reduction::Pca;
//!
//! let dataset = linfa_datasets::iris();
//!
//! // apply PCA projection along a line which maximizes the spread of the data
//! let embedding = Pca::params(1)
//!     .fit(&dataset);
//!
//! // reduce dimensionality of the dataset
//! let dataset = embedding.predict(dataset);
//! ```
//!
use ndarray::{Array1, Array2, ArrayBase, Axis, Data, Ix2};
use ndarray_linalg::{TruncatedOrder, TruncatedSvd};
#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};

use linfa::{
    dataset::Records,
    traits::{Fit, PredictRef},
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
    apply_whitening: bool,
}

impl PcaParams {
    /// Apply whitening to the embedding vector
    ///
    /// Whitening will scale the eigenvalues of the transformation such that the covariance will be
    /// unit diagonal for the original data.
    pub fn whiten(mut self, apply: bool) -> Self {
        self.apply_whitening = apply;

        self
    }
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
/// A fitted PCA model with origin and hyperplane
impl<'a, T> Fit<'a, Array2<f64>, T> for PcaParams {
    type Object = Pca<f64>;

    fn fit(&self, dataset: &DatasetBase<Array2<f64>, T>) -> Pca<f64> {
        let x = dataset.records();
        // calculate mean of data and subtract it
        let mean = x.mean_axis(Axis(0)).unwrap();
        let x = x - &mean;

        // estimate Singular Value Decomposition
        let result = TruncatedSvd::new(x, TruncatedOrder::Largest)
            .decompose(self.embedding_size)
            .unwrap();

        // explained variance is the spectral distribution of the eigenvalues
        let (_, sigma, mut v_t) = result.values_vectors();

        // cut singular values to avoid numerical problems
        let sigma = sigma.mapv(|x| x.max(1e-8));

        // scale the embedding with the square root of the dimensionality and eigenvalue such that
        // the product of the resulting matrix gives the unit covariance.
        if self.apply_whitening {
            let cov_scale = (dataset.nsamples() as f64 - 1.).sqrt();
            for (mut v_t, sigma) in v_t.axis_iter_mut(Axis(0)).zip(sigma.iter()) {
                v_t *= cov_scale / *sigma;
            }
        }

        Pca {
            embedding: v_t,
            sigma,
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
/// use linfa::traits::{Fit, Predict};
/// use linfa_reduction::Pca;
///
/// let dataset = linfa_datasets::iris();
///
/// // apply PCA projection along a line which maximizes the spread of the data
/// let embedding = Pca::params(1)
///     .fit(&dataset);
///
/// // reduce dimensionality of the dataset
/// let dataset = embedding.predict(dataset);
/// ```
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Debug, Clone)]
pub struct Pca<F> {
    embedding: Array2<F>,
    sigma: Array1<F>,
    mean: Array1<F>,
}

impl Pca<f64> {
    /// Create default parameter set
    ///
    /// # Parameters
    ///
    ///  * `embedding_size`: the target dimensionality
    pub fn params(embedding_size: usize) -> PcaParams {
        PcaParams {
            embedding_size,
            apply_whitening: false,
        }
    }

    /// Return the amount of explained variance per element
    pub fn explained_variance(&self) -> Array1<f64> {
        self.sigma.mapv(|x| x * x / (self.sigma.len() as f64 - 1.0))
    }

    /// Return the normalized amount of explained variance per element
    pub fn explained_variance_ratio(&self) -> Array1<f64> {
        let ex_var = self.sigma.mapv(|x| x * x / (self.sigma.len() as f64 - 1.0));
        let sum_ex_var = ex_var.sum();

        ex_var / sum_ex_var
    }

    /// Return the singular values
    pub fn singular_values(&self) -> &Array1<f64> {
        &self.sigma
    }
}

/*
/// Project a matrix to lower dimensional space
///
/// The projection first centers and then projects the data.
impl<F: Float, D: Data<Elem = F>> Predict<ArrayBase<D, Ix2>, Array2<F>> for Pca<F> {
    fn predict(&self, x: ArrayBase<D, Ix2>) -> Array2<F> {
        (&x - &self.mean).dot(&self.embedding.t())
    }
}

/// Project a matrix to lower dimensional space
///
/// The projection first centers and then projects the data.
impl<F: Float, T, D: Data<Elem = F>>
    Predict<DatasetBase<ArrayBase<D, Ix2>, T>, DatasetBase<Array2<F>, T>> for Pca<F>
{
    fn predict(&self, ds: DatasetBase<ArrayBase<D, Ix2>, T>) -> DatasetBase<Array2<F>, T> {
        let new_records = (ds.records() - &self.mean).dot(&self.embedding.t());

        ds.with_records(new_records)
    }
}*/

impl<F: Float, D: Data<Elem = F>> PredictRef<ArrayBase<D, Ix2>, Array2<F>> for Pca<F> {
    fn predict_ref<'a>(&'a self, records: &ArrayBase<D, Ix2>) -> Array2<F> {
        (records - &self.mean).dot(&self.embedding.t())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use linfa::traits::Predict;
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Array2};
    use ndarray_rand::{
        rand_distr::{StandardNormal, Uniform},
        RandomExt,
    };
    use rand::{rngs::SmallRng, SeedableRng};

    /// Small whitening test
    ///
    /// This test rotates 2-dimensional data by 45° and checks whether the whitening transformation
    /// creates a diagonal covariance matrix.
    #[test]
    fn test_whitening_small() {
        // create random number generator
        let mut rng = SmallRng::seed_from_u64(42);

        // rotate data by 45°
        let tmp = Array2::random_using((300, 2), Uniform::new(-1.0f64, 1.), &mut rng);
        let q = array![[1., 1.], [-1., 1.]];

        let dataset = DatasetBase::from(tmp.dot(&q));

        let model = Pca::params(2).whiten(true).fit(&dataset);
        let proj = model.predict(&dataset);

        // check that the covariance is unit diagonal
        let cov = proj.t().dot(&proj);
        assert_abs_diff_eq!(cov / (300. - 1.), Array2::eye(2), epsilon = 1e-5);
    }

    /// Random number whitening test
    ///
    /// This test creates a large number of uniformly distributed random numbers and asserts that
    /// the whitening routine is able to diagonalize the covariance matrix.
    #[test]
    fn test_whitening_rand() {
        // create random number generator
        let mut rng = SmallRng::seed_from_u64(42);

        // generate random data
        let data = Array2::random_using((300, 50), Uniform::new(-1.0f64, 1.), &mut rng);
        let dataset = DatasetBase::from(data);

        let model = Pca::params(10).whiten(true).fit(&dataset);
        let proj = model.predict(&dataset);

        // check that the covariance is unit diagonal
        let cov = proj.t().dot(&proj);
        assert_abs_diff_eq!(cov / (300. - 1.), Array2::eye(10), epsilon = 1e-5);
    }

    /// Eigenvalue structure in high dimensions
    ///
    /// This test checks that the eigenvalues are following the Marchensko-Pastur law. The data is
    /// standard uniformly distributed (i.e. E(x) = 0, E^2(x) = 1) and we have twice the amount of
    /// data when compared to features. The probability density of the eigenvalues should then follow
    /// a special densitiy function, described by the Marchenko-Pastur law.
    ///
    /// See also https://en.wikipedia.org/wiki/Marchenko%E2%80%93Pastur_distribution
    #[test]
    fn test_marchenko_pastur() {
        // create random number generator
        let mut rng = SmallRng::seed_from_u64(3);

        // generate normal distribution random data with N >> p
        let data = Array2::random_using((1000, 500), StandardNormal, &mut rng);
        let dataset = DatasetBase::from(data / 1000f64.sqrt());

        let model = Pca::params(500).fit(&dataset);
        let sv = model.singular_values().mapv(|x| x * x);

        // we have created a random spectrum and can apply the Marchenko-Pastur law
        // with variance 1 and p/n = 0.5
        let (a, b) = (
            1. * (1. - 0.5f64.sqrt()).powf(2.0),
            1. * (1. + 0.5f64.sqrt()).powf(2.0),
        );

        // check that the spectrum has correct boundaries
        assert_abs_diff_eq!(b, sv[0], epsilon = 0.1);
        assert_abs_diff_eq!(a, sv[sv.len() - 1], epsilon = 0.1);

        // estimate density empirical and compare with Marchenko-Pastur law
        let mut i = 0;
        'outer: for th in Array1::linspace(0.1, 2.8, 28).into_iter().rev() {
            let mut count = 0;
            while sv[i] >= *th {
                count += 1;
                i += 1;

                if i == sv.len() {
                    break 'outer;
                }
            }

            let x = th + 0.05;
            let mp_law = ((b - x) * (x - a)).sqrt() / std::f64::consts::PI / x;
            let empirical = count as f64 / 500. / ((2.8 - 0.1) / 28.);

            assert_abs_diff_eq!(mp_law, empirical, epsilon = 0.05);
        }
    }

    #[test]
    fn test_explained_variance_cutoff() {
        // create random number generator
        let mut rng = SmallRng::seed_from_u64(42);

        // generate high dimensional data with two orthogonal vectors
        let n = 500;
        let mut a = Array1::<f64>::random_using(n, StandardNormal, &mut rng);
        a /= (a.t().dot(&a)).sqrt();

        // perform a single step of the Gram-Schmidt process
        let mut b = Array1::random_using(n, StandardNormal, &mut rng);
        b -= &(b.t().dot(&a) * &a);
        b /= (b.t().dot(&b)).sqrt();

        // construct matrix with rank 2
        let data =
            Array2::from_shape_fn((500, 500), |dim| a[dim.0] * a[dim.1] + b[dim.0] * b[dim.1]);

        let dataset = DatasetBase::from(data);

        // fit PCA with 10 possible embeddings
        let model = Pca::params(10).fit(&dataset);

        // only two eigenvalues are relevant
        assert_eq!(model.explained_variance_ratio().len(), 2);
        // both of them explain approximately the same variance
        assert_abs_diff_eq!(
            model.explained_variance_ratio(),
            array![1. / 2., 1. / 2.],
            epsilon = 1e-2
        );
    }

    #[test]
    fn test_explained_variance_diag() {
        let dataset = DatasetBase::from(Array2::from_diag(&array![1., 1., 1., 1.]));
        let model = Pca::params(3).fit(&dataset);

        assert_abs_diff_eq!(
            model.explained_variance_ratio(),
            array![1. / 3., 1. / 3., 1. / 3.],
            epsilon = 1e-6
        );
    }
}
