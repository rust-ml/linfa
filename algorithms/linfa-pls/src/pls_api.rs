use crate::algorithm::*;
use crate::errors;
use linfa::{traits::Fit, DatasetBase, Float};
use ndarray::{Array1, Array2, ArrayBase, Data, Ix2};
use ndarray_linalg::{Lapack, Scalar};

pub use errors::*;

macro_rules! pls_algo { ($name:ident) => {
    paste::item! {

        pub struct [<Pls $name Params>]<F: Float>(PlsParams<F>);

        impl<F: Float> [<Pls $name Params>]<F> {
            pub fn max_iterations(mut self, max_iter: usize) -> Self {
                self.0.max_iter = max_iter;
                self
            }
            pub fn tolerance(mut self, tolerance: F) -> Self {
                self.0.tolerance = tolerance;
                self
            }
            pub fn scale(mut self, scale: bool) -> Self {
                self.0.scale = scale;
                self
            }
            pub fn algorithm(mut self, algorithm: Algorithm) -> Self {
                self.0.algorithm = algorithm;
                self
            }
        }

        pub struct [<Pls $name>]<F: Float>(Pls<F>);
        impl<F: Float> [<Pls $name>]<F> {
            pub fn default() -> [<Pls $name Params>]<F> {
                Self::params(2)
            }
            pub fn params(n_components: usize) -> [<Pls $name Params>]<F> {
                [<Pls $name Params>](Pls::[<$name:lower>](n_components))
            }
            pub fn means(&self) -> (&Array1<F>, &Array1<F>) {
                self.0.means()
            }
            pub fn stds(&self) -> (&Array1<F>, &Array1<F>) {
                self.0.stds()
            }
            pub fn weights(&self) -> (&Array2<F>, &Array2<F>) {
                self.0.weights()
            }
            pub fn scores(&self) -> (&Array2<F>, &Array2<F>) {
                self.0.scores()
            }
            pub fn loadings(&self) -> (&Array2<F>, &Array2<F>) {
                self.0.loadings()
            }
            pub fn rotations(&self) -> (&Array2<F>, &Array2<F>) {
                self.0.rotations()
            }
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

    }
}}

pls_algo!(Regression);
pls_algo!(Canonical);
pls_algo!(Cca);
