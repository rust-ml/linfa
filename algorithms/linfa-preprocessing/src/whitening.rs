//! Methods for uncorrelating data
//!
//! Whitening refers to a collection of methods that, given in input a matrix `X` of records with
//! covariance matrix =  `sigma`, output a whitening matrix `W` such that `W.T`*`W` = `sigma`.
//! Appliyng the whitening matrix `W` to the input data gives a new data matrix `Y` such that `Y` has
//! unit diagonal (white) covariance matrix.

use crate::error::{Error, Result};
use crate::Float;
use linfa::dataset::AsTargets;
use linfa::dataset::Records;
use linfa::traits::{Fit, Transformer};
use linfa::DatasetBase;
use ndarray::{Array1, Array2, ArrayBase, ArrayView1, ArrayView2, Axis, Data, Ix2};
use ndarray_linalg::cholesky::{Cholesky, UPLO};
use ndarray_linalg::solve::Inverse;
use ndarray_linalg::svd::SVD;

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

impl<'a, F: Float + approx::AbsDiffEq, D: Data<Elem = F>, T: AsTargets>
    Fit<'a, ArrayBase<D, Ix2>, T> for Whitener
{
    type Object = Result<FittedWhitener<F>>;

    fn fit(&self, x: &DatasetBase<ArrayBase<D, Ix2>, T>) -> Self::Object {
        if x.records().dim().0 == 0 {
            return Err(Error::NotEnoughSamples);
        }
        let mean = x.records().mean_axis(Axis(0)).unwrap();
        let sigma = x.records() - &mean;
        match self.method {
            WhiteningMethod::Pca => {
                let (_, s, v_t) = sigma.svd(false, true)?;
                if v_t.is_none() {
                    return Err(Error::NoneEignenvectors);
                }
                let mut v_t = v_t.unwrap();
                let s = s.mapv(|x| F::from(x).unwrap().max(F::from(1e-8).unwrap()));
                let cov_scale = (F::from(x.nsamples()).unwrap() - F::one()).sqrt();
                for (mut v_t, s) in v_t.axis_iter_mut(Axis(0)).zip(s.iter()) {
                    v_t *= cov_scale / *s;
                }
                Ok(FittedWhitener {
                    transformation_matrix: v_t,
                    mean,
                })
            }
            WhiteningMethod::Zca => {
                let sigma = sigma.t().dot(&sigma) / F::from(x.nsamples() - 1).unwrap();
                let (u, s, _) = sigma.svd(true, false)?;
                if u.is_none() {
                    return Err(Error::NoneEignenvectors);
                }
                let u = u.unwrap();
                let s =
                    s.mapv(|x| (F::one() / F::from(x).unwrap().sqrt()).max(F::from(1e-8).unwrap()));
                let lambda: Array2<F> = Array2::<F>::eye(s.len()) * s;
                let transformation_matrix = u.dot(&lambda).dot(&u.t());
                Ok(FittedWhitener {
                    transformation_matrix,
                    mean,
                })
            }
            WhiteningMethod::Cholesky => {
                let sigma = sigma.t().dot(&sigma) / F::from(x.nsamples() - 1).unwrap();
                let transformation_matrix = sigma.inv()?.cholesky(UPLO::Upper)?;
                Ok(FittedWhitener {
                    transformation_matrix,
                    mean,
                })
            }
        }
    }
}

/// Struct that can be used to whiten data. Data will be scaled according to the whitening matrix learned
/// during fitting.
/// Obtained by fitting a [Whitener](struct.Whitener.html).
///
/// Transforming the data used during fitting will yield a scaled data matrix with
/// unit diagonal covariance matrix.
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
        return sigma;
    }

    fn inv_cov<D: Data<Elem = f64>>(x: &ArrayBase<D, Ix2>) -> Array2<f64> {
        return cov(x).inv().unwrap();
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
        let whitener = Whitener::pca().fit(&train).unwrap();
        let _whitened_train = whitener.transform(train);
        let _whitened_val = whitener.transform(val);
    }
}
