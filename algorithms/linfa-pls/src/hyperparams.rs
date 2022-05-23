use crate::{Algorithm, DeflationMode, Mode, PlsError};
use linfa::{Float, ParamGuard};

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct PlsValidParams<F: Float> {
    n_components: usize,
    max_iter: usize,
    tolerance: F,
    scale: bool,
    algorithm: Algorithm,
    deflation_mode: DeflationMode,
    mode: Mode,
}

impl<F: Float> PlsValidParams<F> {
    pub fn n_components(&self) -> usize {
        self.n_components
    }

    pub fn max_iter(&self) -> usize {
        self.max_iter
    }

    pub fn tolerance(&self) -> F {
        self.tolerance
    }

    pub fn scale(&self) -> bool {
        self.scale
    }

    pub fn algorithm(&self) -> Algorithm {
        self.algorithm
    }

    pub fn deflation_mode(&self) -> DeflationMode {
        self.deflation_mode
    }

    pub fn mode(&self) -> Mode {
        self.mode
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct PlsParams<F: Float>(pub(crate) PlsValidParams<F>);

impl<F: Float> PlsParams<F> {
    pub fn new(n_components: usize) -> PlsParams<F> {
        Self(PlsValidParams {
            n_components,
            max_iter: 500,
            tolerance: F::cast(1e-6),
            scale: true,
            algorithm: Algorithm::Nipals,
            deflation_mode: DeflationMode::Regression,
            mode: Mode::A,
        })
    }

    #[cfg(test)]
    pub fn max_iterations(mut self, max_iter: usize) -> Self {
        self.0.max_iter = max_iter;
        self
    }

    #[cfg(test)]
    pub fn tolerance(mut self, tolerance: F) -> Self {
        self.0.tolerance = tolerance;
        self
    }

    #[cfg(test)]
    pub fn scale(mut self, scale: bool) -> Self {
        self.0.scale = scale;
        self
    }

    #[cfg(test)]
    pub fn algorithm(mut self, algorithm: Algorithm) -> Self {
        self.0.algorithm = algorithm;
        self
    }

    pub fn deflation_mode(mut self, deflation_mode: DeflationMode) -> Self {
        self.0.deflation_mode = deflation_mode;
        self
    }

    pub fn mode(mut self, mode: Mode) -> Self {
        self.0.mode = mode;
        self
    }
}

impl<F: Float> ParamGuard for PlsParams<F> {
    type Checked = PlsValidParams<F>;
    type Error = PlsError;

    fn check_ref(&self) -> Result<&Self::Checked, Self::Error> {
        if self.0.tolerance.is_negative()
            || self.0.tolerance.is_nan()
            || self.0.tolerance.is_infinite()
        {
            Err(PlsError::InvalidTolerance(
                self.0.tolerance().to_f32().unwrap(),
            ))
        } else if self.0.max_iter == 0 {
            Err(PlsError::ZeroMaxIter)
        } else {
            Ok(&self.0)
        }
    }

    fn check(self) -> Result<Self::Checked, Self::Error> {
        self.check_ref()?;
        Ok(self.0)
    }
}

macro_rules! pls_algo { ($name:ident) => {
    paste::item! {
        pub struct [<Pls $name Params>]<F: Float>(pub(crate) [<Pls $name ValidParams>]<F>);
        pub struct [<Pls $name ValidParams>]<F: Float>(pub(crate) PlsValidParams<F>);

        impl<F: Float> [<Pls $name Params>]<F> {
            /// Set the maximum number of iterations of the power method when algorithm='Nipals'. Ignored otherwise.
            pub fn max_iterations(mut self, max_iter: usize) -> Self {
                self.0.0.max_iter = max_iter;
                self
            }

            /// Set the tolerance used as convergence criteria in the power method: the algorithm
            /// stops whenever the squared norm of u_i - u_{i-1} is less than tol, where u corresponds
            /// to the left singular vector.
            pub fn tolerance(mut self, tolerance: F) -> Self {
                self.0.0.tolerance = tolerance;
                self
            }

            /// Set whether to scale the dataset
            pub fn scale(mut self, scale: bool) -> Self {
                self.0.0.scale = scale;
                self
            }

            /// Set the algorithm used to estimate the first singular vectors of the cross-covariance matrix.
            /// `Nipals` uses the power method while `Svd` will compute the whole SVD.
            pub fn algorithm(mut self, algorithm: Algorithm) -> Self {
                self.0.0.algorithm = algorithm;
                self
            }
        }

        impl<F: Float> ParamGuard for [<Pls $name Params>]<F> {
            type Checked = [<Pls $name ValidParams>]<F>;
            type Error = PlsError;

            fn check_ref(&self) -> Result<&Self::Checked, Self::Error> {
                if self.0.0.tolerance.is_negative() || self.0.0.tolerance.is_nan() || self.0.0.tolerance.is_infinite() {
                    Err(PlsError::InvalidTolerance(self.0.0.tolerance.to_f32().unwrap()))
                } else if self.0.0.max_iter == 0 {
                    Err(PlsError::ZeroMaxIter)
                } else {
                    Ok(&self.0)
                }
            }

            fn check(self) -> Result<Self::Checked, Self::Error> {
                self.check_ref()?;
                Ok(self.0)
            }
        }
    }
}}

pls_algo!(Regression);
pls_algo!(Canonical);
pls_algo!(Cca);
