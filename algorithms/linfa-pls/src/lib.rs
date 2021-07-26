//! # Partial Least Squares
//!
//! `linfa-pls` provides an implementation of methods in the PLS (Partial Least Squares) family.
//! The PLS method is a statistical method that finds a linear relationship between
//! input variables and output variables by projecting them onto a new subspace formed
//! by newly chosen variables (aka latent variables), which are linear
//! combinations of the input variables. The subspace is choosen to maximize the
//! covariance between responses and independant variables.
//!
//! This approach is particularly useful when the original data are characterized by
//! a large number of highly collinear variables measured on a small number of samples.
//!
//! The implementation is a port of the scikit-learn 0.24 cross-decomposition module.
//!
//! ## References
//!
//! * [A survey of Partial Least Squares (PLS) methods, with emphasis on the two-block case JA Wegelin](https://stat.uw.edu/sites/default/files/files/reports/2000/tr371.pdf)
//! * [Scikit-Learn User Guide](https://scikit-learn.org/stable/modules/cross_decomposition.html#cross-decomposition)
//!
//! ## Example
//!
//! ```rust, ignore
//! use linfa::prelude::*;
//! use linfa_pls::{errors::Result, PlsRegression};
//! use ndarray::array;
//!
//! // Load linnerud datase 20 samples, 3 input features, 3 output features
//! let ds = linnerud();
//!
//! // Fit PLS2 method using 2 principal components (latent variables)
//! let pls = PlsRegression::params(2).fit(&ds)?;
//!
//! // We can either apply the dimension reduction to a dataset
//! let reduced_ds = pls.transform(ds);
//!
//! // ... or predict outputs given a new input sample.
//! let exercices = array![[14., 146., 61.], [6., 80., 60.]];
//! let physio_measures = pls.predict(exercices);
//! ```
mod errors;
mod pls_generic;
mod pls_svd;
mod utils;

use crate::pls_generic::*;

use linfa::{traits::Fit, traits::PredictInplace, traits::Transformer, DatasetBase};
use ndarray::{Array2, ArrayBase, Data, Ix2};
use ndarray_linalg::{Lapack, Scalar};

pub use errors::*;
pub use pls_svd::*;

/// Add Scalar and Lapack trait bounds to the common Float trait
pub trait Float: linfa::Float + Scalar + Lapack {}

impl Float for f32 {}
impl Float for f64 {}

macro_rules! pls_algo { ($name:ident) => {
    paste::item! {

        pub struct [<Pls $name Params>]<F: Float>(PlsParams<F>);

        impl<F: Float> [<Pls $name Params>]<F> {
            /// Set the maximum number of iterations of the power method when algorithm='Nipals'. Ignored otherwise.
            pub fn max_iterations(mut self, max_iter: usize) -> Self {
                self.0.max_iter = max_iter;
                self
            }

            /// Set the tolerance used as convergence criteria in the power method: the algorithm
            /// stops whenever the squared norm of u_i - u_{i-1} is less than tol, where u corresponds
            /// to the left singular vector.
            pub fn tolerance(mut self, tolerance: F) -> Self {
                self.0.tolerance = tolerance;
                self
            }

            /// Set whether to scale the dataset
            pub fn scale(mut self, scale: bool) -> Self {
                self.0.scale = scale;
                self
            }

            /// Set the algorithm used to estimate the first singular vectors of the cross-covariance matrix.
            /// `Nipals` uses the power method while `Svd` will compute the whole SVD.
            pub fn algorithm(mut self, algorithm: Algorithm) -> Self {
                self.0.algorithm = algorithm;
                self
            }
        }

        pub struct [<Pls $name>]<F: Float>(Pls<F>);
        impl<F: Float> [<Pls $name>]<F> {

            pub fn params(n_components: usize) -> [<Pls $name Params>]<F> {
                [<Pls $name Params>](Pls::[<$name:lower>](n_components))
            }

            /// Singular vectors of the cross-covariance matrices
            pub fn weights(&self) -> (&Array2<F>, &Array2<F>) {
                self.0.weights()
            }

            /// Loadings of records and targets
            pub fn loadings(&self) -> (&Array2<F>, &Array2<F>) {
                self.0.loadings()
            }

            /// Projection matrices used to transform records and targets
            pub fn rotations(&self) -> (&Array2<F>, &Array2<F>) {
                self.0.rotations()
            }

            /// The coefficients of the linear model such that Y is approximated as Y = X.coefficients
            pub fn coefficients(&self) -> &Array2<F> {
                self.0.coefficients()
            }

            /// Transform the given dataset in the projected space back to the original space.
            pub fn inverse_transform(
                &self,
                dataset: DatasetBase<
                    ArrayBase<impl Data<Elem = F>, Ix2>,
                    ArrayBase<impl Data<Elem = F>, Ix2>,
                >,
            ) -> DatasetBase<Array2<F>, Array2<F>> {
                self.0.inverse_transform(dataset)
            }
        }

        impl<F: Float, D: Data<Elem = F>> Fit<ArrayBase<D, Ix2>, ArrayBase<D, Ix2>, PlsError>
            for [<Pls $name Params>]<F>
        {
            type Object = [<Pls $name>]<F>;
            fn fit(
                &self,
                dataset: &DatasetBase<ArrayBase<D, Ix2>, ArrayBase<D, Ix2>>,
            ) -> Result<Self::Object> {
                let pls = self.0.fit(dataset)?;
                Ok([<Pls $name>](pls))
            }
        }

        impl<F: Float, D: Data<Elem = F>> Transformer<
            DatasetBase<ArrayBase<D, Ix2>, ArrayBase<D, Ix2>>,
            DatasetBase<Array2<F>, Array2<F>>,
        > for [<Pls $name>]<F>
        {
            /// Apply dimension reduction to the given dataset
            fn transform(
                &self,
                dataset: DatasetBase<ArrayBase<D, Ix2>, ArrayBase<D, Ix2>>,
            ) -> DatasetBase<Array2<F>, Array2<F>> {
                self.0.transform(dataset)
            }
        }

        impl<F: Float, D: Data<Elem = F>> PredictInplace<ArrayBase<D, Ix2>, Array2<F>> for [<Pls $name>]<F> {
            /// Given an input matrix `X`, with shape `(n_samples, n_features)`,
            /// `predict` returns the target variable according to [<Pls $name>] method
            /// learned from the training data distribution.
            fn predict_inplace<'a>(&'a self, x: &ArrayBase<D, Ix2>, y: &mut Array2<F>) {
                assert_eq!(
                    y.shape(),
                    &[x.nrows(), PredictInplace::<Array2<_>, _>::num_target_variables_hint(self)],
                    "The number of data points must match the number of output targets."
                );

                self.0.predict_inplace(x, y);
            }

            fn num_target_variables_hint(&self) -> usize {
                PredictInplace::<ArrayBase<D, Ix2>, Array2<F>>::num_target_variables_hint(&self.0)
            }
        }
    }
}}

pls_algo!(Regression);
pls_algo!(Canonical);
pls_algo!(Cca);

#[cfg(test)]
mod test {
    use super::*;
    use approx::assert_abs_diff_eq;
    use linfa::{traits::Fit, traits::Predict, traits::Transformer};
    use linfa_datasets::linnerud;
    use ndarray::array;

    macro_rules! test_pls_algo {
        (Svd) => {
            paste::item! {
                #[test]
                fn [<test_pls_svd>]() -> Result<()> {
                    let ds = linnerud();
                    let pls = PlsSvd::<f64>::params(3).fit(&ds)?;
                    let _ds1 = pls.transform(ds);
                    Ok(())
                }
            }
        };

        ($name:ident, $expected:expr) => {
            paste::item! {
                #[test]
                fn [<test_pls_$name:lower>]() -> Result<()> {
                    let ds = linnerud();
                    let pls = [<Pls $name>]::<f64>::params(2).fit(&ds)?;
                    let _ds1 = pls.transform(ds);
                    let exercices = array![[14., 146., 61.], [6., 80., 60.]];
                    let physios = pls.predict(exercices);
                    assert_abs_diff_eq!($expected, physios.targets(), epsilon=1e-2);
                    Ok(())
                }
            }
        };
    }

    // Prediction values were checked against scikit-learn 0.24.1
    test_pls_algo!(
        Canonical,
        array![
            [180.56979423, 33.29543984, 56.90850758],
            [190.854022, 38.91963398, 53.26914489]
        ]
    );
    test_pls_algo!(
        Regression,
        array![
            [172.39580643, 34.11919145, 57.15430526],
            [192.11167813, 38.05058858, 53.99844922]
        ]
    );
    test_pls_algo!(
        Cca,
        array![
            [181.56238421, 34.42502589, 57.31447865],
            [205.11767414, 40.23445194, 52.26494323]
        ]
    );
    test_pls_algo!(Svd);

    #[test]
    fn test_one_component_equivalence() -> Result<()> {
        // PlsRegression, PlsSvd and PLSCanonical should all be equivalent when n_components is 1
        let ds = linnerud();
        let regression = PlsRegression::params(1).fit(&ds)?.transform(linnerud());
        let canonical = PlsCanonical::params(1).fit(&ds)?.transform(linnerud());
        let svd = PlsSvd::<f64>::params(1).fit(&ds)?.transform(linnerud());

        assert_abs_diff_eq!(regression.records(), canonical.records(), epsilon = 1e-5);
        assert_abs_diff_eq!(svd.records(), canonical.records(), epsilon = 1e-5);
        Ok(())
    }
}
