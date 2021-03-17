//! # Partial Least Squares
//!
//! `linfa-pls` provides an implementation of Partial Least Squares family methods.
//! The PLS method is a statistical method that finds a linear relationship between
//! input variables and output variables by projecting input variables onto a new
//! subspace formed by newly chosen variables (aka latent variables), which are linear
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

use linfa::{
    traits::Fit, traits::Predict, traits::PredictRef, traits::Transformer, DatasetBase, Float,
};
use ndarray::{Array2, ArrayBase, Data, Ix2};
use ndarray_linalg::{Lapack, Scalar};

pub use errors::*;
pub use pls_svd::*;

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
        }

        impl<F: Float + Scalar + Lapack, D: Data<Elem = F>> Fit<'_, ArrayBase<D, Ix2>, ArrayBase<D, Ix2>>
            for [<Pls $name Params>]<F>
        {
            type Object = Result<[<Pls $name>]<F>>;
            fn fit(
                &self,
                dataset: &DatasetBase<ArrayBase<D, Ix2>, ArrayBase<D, Ix2>>,
            ) -> Result<[<Pls $name>]<F>> {
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

        impl<F: Float, D: Data<Elem = F>> PredictRef<ArrayBase<D, Ix2>, Array2<F>> for [<Pls $name>]<F> {
            /// Given an input matrix `X`, with shape `(n_samples, n_features)`,
            /// `predict` returns the target variable according to [<Pls $name>] method
            /// learned from the training data distribution.
            fn predict_ref<'a>(&'a self, x: &ArrayBase<D, Ix2>) -> Array2<F> {
                self.0.predict_ref(x)
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

        ($name:ident) => {
            paste::item! {
                #[test]
                fn [<test_pls_$name:lower>]() -> Result<()> {
                    let ds = linnerud();
                    let pls = [<Pls $name>]::<f64>::params(3).fit(&ds)?;
                    let _ds1 = pls.transform(ds);
                    let exercices = array![[14., 146., 61.], [6., 80., 60.]];
                    let physios = pls.predict(exercices);
                    println!("Physiologicals = {:?}", physios.targets());
                    Ok(())
                }
            }
        };
    }

    test_pls_algo!(Canonical);
    test_pls_algo!(Regression);
    test_pls_algo!(Cca);
    test_pls_algo!(Svd);
}
