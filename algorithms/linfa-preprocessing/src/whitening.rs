//! Methods for uncorrelating data
//!
//! Whitening refers to a collection of methods that, given in input a matrix `X` of records with
//! covariance matrix =  `sigma`, output a whitening matrix `W` such that `W.T` dot `W` = `sigma`.
//! Appliyng the whitening matrix `W` to the input data gives a new data matrix `Y` of the same
//! size as the input such that `Y` has
//! unit diagonal (white) covariance matrix.

use crate::error::{PreprocessingError, Result};
use linfa::dataset::{AsTargets, Records, WithLapack, WithoutLapack};
use linfa::traits::{Fit, Transformer};
use linfa::{DatasetBase, Float};
#[cfg(not(feature = "blas"))]
use linfa_linalg::{
    cholesky::{CholeskyInplace, InverseCInplace},
    svd::SVD,
};
use ndarray::{Array1, Array2, ArrayBase, ArrayView1, ArrayView2, Axis, Data, Ix2};
#[cfg(feature = "blas")]
use ndarray_linalg::{
    cholesky::{CholeskyInto, InverseCInto, UPLO},
    svd::SVD,
    Scalar,
};

pub enum WhiteningMethod {
    Pca,
    Zca,
    Cholesky,
}

/// Struct that can be fitted to the input data to obtain the related whitening matrix.
/// Fitting returns a [FittedWhitener](struct.FittedWhitener.html) struct that can be used to
/// apply the whitening transformation to the input data.
pub struct Whitener {
    method: WhiteningMethod,
}

impl Whitener {
    /// Creates an instance of a Whitener that uses the PCA method
    pub fn pca() -> Self {
        Self {
            method: WhiteningMethod::Pca,
        }
    }
    /// Creates an instance of a Whitener that uses the ZCA (Mahalanobis) method
    pub fn zca() -> Self {
        Self {
            method: WhiteningMethod::Zca,
        }
    }
    /// Creates an instance of a Whitener that uses the cholesky decomposition of the inverse of the covariance matrix
    pub fn cholesky() -> Self {
        Self {
            method: WhiteningMethod::Cholesky,
        }
    }

    pub fn method(mut self, method: WhiteningMethod) -> Self {
        self.method = method;
        self
    }
}

impl<F: Float, D: Data<Elem = F>, T: AsTargets> Fit<ArrayBase<D, Ix2>, T, PreprocessingError>
    for Whitener
{
    type Object = FittedWhitener<F>;

    fn fit(&self, x: &DatasetBase<ArrayBase<D, Ix2>, T>) -> Result<Self::Object> {
        if x.nsamples() == 0 {
            return Err(PreprocessingError::NotEnoughSamples);
        }
        // safe because of above zero samples check
        let mean = x.records().mean_axis(Axis(0)).unwrap();
        let sigma = x.records() - &mean;

        // add Lapack + Scalar trait bounds
        let sigma = sigma.with_lapack();

        let transformation_matrix = match self.method {
            WhiteningMethod::Pca => {
                let (_, s, v_t) = sigma.svd(false, true)?;

                // Safe because the second argument in the above call is set to true
                let mut v_t = v_t.unwrap().without_lapack();
                #[cfg(feature = "blas")]
                let s = s.mapv(Scalar::from_real);
                let s = s.without_lapack();

                let s = s.mapv(|x: F| x.max(F::cast(1e-8)));

                let cov_scale = F::cast(x.nsamples() - 1).sqrt();
                for (mut v_t, s) in v_t.axis_iter_mut(Axis(0)).zip(s.iter()) {
                    v_t *= cov_scale / *s;
                }

                v_t
            }
            WhiteningMethod::Zca => {
                let sigma = sigma.t().dot(&sigma) / F::Lapack::cast(x.nsamples() - 1);
                let (u, s, _) = sigma.svd(true, false)?;

                // Safe because the first argument in the above call is set to true
                let u = u.unwrap().without_lapack();
                #[cfg(feature = "blas")]
                let s = s.mapv(Scalar::from_real);
                let s = s.without_lapack();

                let s = s.mapv(|x: F| (F::one() / x.sqrt()).max(F::cast(1e-8)));
                let lambda: Array2<F> = Array2::<F>::eye(s.len()) * s;
                u.dot(&lambda).dot(&u.t())
            }
            WhiteningMethod::Cholesky => {
                let sigma = sigma.t().dot(&sigma) / F::Lapack::cast(x.nsamples() - 1);
                // sigma must be positive definite for us to call cholesky on its inverse, so invc
                // is allowed here
                #[cfg(feature = "blas")]
                let out = sigma
                    .invc_into()?
                    .cholesky_into(UPLO::Upper)?
                    .without_lapack();
                #[cfg(not(feature = "blas"))]
                let mut sigma = sigma;
                #[cfg(not(feature = "blas"))]
                let out = sigma
                    .invc_inplace()?
                    .reversed_axes()
                    .cholesky_into()?
                    .reversed_axes()
                    .without_lapack();
                out
            }
        };

        Ok(FittedWhitener {
            transformation_matrix,
            mean,
        })
    }
}

/// Struct that can be used to whiten data. Data will be scaled according to the whitening matrix learned
/// during fitting.
/// Obtained by fitting a [Whitener](struct.Whitener.html).
///
/// Transforming the data used during fitting will yield a scaled data matrix with
/// unit diagonal covariance matrix.
///
/// ### Example
///
/// ```rust
/// use linfa::traits::{Fit, Transformer};
/// use linfa_preprocessing::whitening::Whitener;
///
/// // Load dataset
/// let dataset = linfa_datasets::diabetes();
/// // Learn whitening parameters
/// let whitener = Whitener::pca().fit(&dataset).unwrap();
/// // transform dataset according to whitening parameters
/// let dataset = whitener.transform(dataset);
/// ```
pub struct FittedWhitener<F: Float> {
    transformation_matrix: Array2<F>,
    mean: Array1<F>,
}

impl<F: Float> FittedWhitener<F> {
    /// The matrix used for scaling the data
    pub fn transformation_matrix(&self) -> ArrayView2<F> {
        self.transformation_matrix.view()
    }

    /// The means that will be subtracted to the features before scaling the data
    pub fn mean(&self) -> ArrayView1<F> {
        self.mean.view()
    }
}

impl<F: Float> Transformer<Array2<F>, Array2<F>> for FittedWhitener<F> {
    fn transform(&self, x: Array2<F>) -> Array2<F> {
        (x - &self.mean).dot(&self.transformation_matrix.t())
    }
}

impl<F: Float, D: Data<Elem = F>, T: AsTargets>
    Transformer<DatasetBase<ArrayBase<D, Ix2>, T>, DatasetBase<Array2<F>, T>>
    for FittedWhitener<F>
{
    fn transform(&self, x: DatasetBase<ArrayBase<D, Ix2>, T>) -> DatasetBase<Array2<F>, T> {
        let feature_names = x.feature_names();
        let (records, targets, weights) = (x.records, x.targets, x.weights);
        let records = self.transform(records.to_owned());
        DatasetBase::new(records, targets)
            .with_weights(weights)
            .with_feature_names(feature_names)
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use approx::assert_abs_diff_eq;

    use ndarray_rand::{
        rand::distributions::Uniform, rand::rngs::SmallRng, rand::SeedableRng, RandomExt,
    };

    fn cov<D: Data<Elem = f64>>(x: &ArrayBase<D, Ix2>) -> Array2<f64> {
        let mean = x.mean_axis(Axis(0)).unwrap();
        let sigma = x - &mean;
        let sigma = sigma.t().dot(&sigma) / ((x.dim().0 - 1) as f64);
        sigma
    }

    fn inv_cov<D: Data<Elem = f64>>(x: &ArrayBase<D, Ix2>) -> Array2<f64> {
        #[cfg(feature = "blas")]
        let inv = cov(x).invc_into().unwrap();
        #[cfg(not(feature = "blas"))]
        let inv = cov(x).invc_inplace().unwrap();
        inv
    }

    #[test]
    fn test_zca_matrix() {
        let mut rng = SmallRng::seed_from_u64(42);
        let dataset = Array2::random_using((1000, 7), Uniform::from(-30. ..30.), &mut rng).into();
        let whitener = Whitener::zca().fit(&dataset).unwrap();
        let inv_cov_est = whitener
            .transformation_matrix()
            .t()
            .dot(&whitener.transformation_matrix());
        let inv_cov = inv_cov(dataset.records());
        assert_abs_diff_eq!(inv_cov, inv_cov_est, epsilon = 1e-9);
    }

    #[test]
    fn test_cholesky_matrix() {
        let mut rng = SmallRng::seed_from_u64(42);
        let dataset = Array2::random_using((1000, 7), Uniform::from(-30. ..30.), &mut rng).into();
        let whitener = Whitener::cholesky().fit(&dataset).unwrap();
        let inv_cov_est = whitener
            .transformation_matrix()
            .t()
            .dot(&whitener.transformation_matrix());
        let inv_cov = inv_cov(dataset.records());
        assert_abs_diff_eq!(inv_cov, inv_cov_est, epsilon = 1e-10);
    }

    #[test]
    fn test_pca_matrix() {
        let mut rng = SmallRng::seed_from_u64(42);
        let dataset = Array2::random_using((1000, 7), Uniform::from(-30. ..30.), &mut rng).into();
        let whitener = Whitener::pca().fit(&dataset).unwrap();
        let inv_cov_est = whitener
            .transformation_matrix()
            .t()
            .dot(&whitener.transformation_matrix());
        let inv_cov = inv_cov(dataset.records());
        assert_abs_diff_eq!(inv_cov, inv_cov_est, epsilon = 1e-10);
    }

    #[test]
    fn test_cholesky_whitening() {
        let mut rng = SmallRng::seed_from_u64(64);
        let dataset = Array2::random_using((1000, 7), Uniform::from(-30. ..30.), &mut rng).into();
        let whitener = Whitener::cholesky().fit(&dataset).unwrap();
        let whitened = whitener.transform(dataset);
        let cov = cov(whitened.records());
        assert_abs_diff_eq!(cov, Array2::eye(cov.dim().0), epsilon = 1e-10)
    }

    #[test]
    fn test_zca_whitening() {
        let mut rng = SmallRng::seed_from_u64(64);
        let dataset = Array2::random_using((1000, 7), Uniform::from(-30. ..30.), &mut rng).into();
        let whitener = Whitener::zca().fit(&dataset).unwrap();
        let whitened = whitener.transform(dataset);
        let cov = cov(whitened.records());
        assert_abs_diff_eq!(cov, Array2::eye(cov.dim().0), epsilon = 1e-10)
    }

    #[test]
    fn test_pca_whitening() {
        let mut rng = SmallRng::seed_from_u64(64);
        let dataset = Array2::random_using((1000, 7), Uniform::from(-30. ..30.), &mut rng).into();
        let whitener = Whitener::pca().fit(&dataset).unwrap();
        let whitened = whitener.transform(dataset);
        let cov = cov(whitened.records());
        assert_abs_diff_eq!(cov, Array2::eye(cov.dim().0), epsilon = 1e-10)
    }

    #[test]
    fn test_train_val_matrix() {
        let (train, val) = linfa_datasets::diabetes().split_with_ratio(0.9);
        let (train_dim, val_dim) = (train.records().dim(), val.records().dim());
        let whitener = Whitener::pca().fit(&train).unwrap();
        let whitened_train = whitener.transform(train);
        let whitened_val = whitener.transform(val);
        assert_eq!(train_dim, whitened_train.records.dim());
        assert_eq!(val_dim, whitened_val.records.dim());
    }

    #[test]
    fn test_retain_feature_names() {
        let dataset = linfa_datasets::diabetes();
        let original_feature_names = dataset.feature_names();
        let transformed = Whitener::cholesky()
            .fit(&dataset)
            .unwrap()
            .transform(dataset);
        assert_eq!(original_feature_names, transformed.feature_names())
    }

    #[test]
    #[should_panic]
    fn test_pca_fail_on_empty_input() {
        let dataset: DatasetBase<Array2<f64>, _> = Array2::zeros((0, 0)).into();
        let _whitener = Whitener::pca().fit(&dataset).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_zca_fail_on_empty_input() {
        let dataset: DatasetBase<Array2<f64>, _> = Array2::zeros((0, 0)).into();
        let _whitener = Whitener::zca().fit(&dataset).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_cholesky_fail_on_empty_input() {
        let dataset: DatasetBase<Array2<f64>, _> = Array2::zeros((0, 0)).into();
        let _whitener = Whitener::cholesky().fit(&dataset).unwrap();
    }
}
