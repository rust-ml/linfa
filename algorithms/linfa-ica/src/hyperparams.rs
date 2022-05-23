use crate::{error::FastIcaError, fast_ica::FastIca, fast_ica::GFunc};
use linfa::{Float, ParamGuard};
#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};

/// Fast Independent Component Analysis (ICA)
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq)]
pub struct FastIcaValidParams<F: Float> {
    ncomponents: Option<usize>,
    gfunc: GFunc,
    max_iter: usize,
    tol: F,
    random_state: Option<usize>,
}

impl<F: Float> FastIcaValidParams<F> {
    pub fn ncomponents(&self) -> &Option<usize> {
        &self.ncomponents
    }

    pub fn gfunc(&self) -> &GFunc {
        &self.gfunc
    }

    pub fn max_iter(&self) -> usize {
        self.max_iter
    }

    pub fn tol(&self) -> F {
        self.tol
    }

    pub fn random_state(&self) -> &Option<usize> {
        &self.random_state
    }
}

#[derive(Debug, Copy, Clone, PartialOrd, PartialEq)]
pub struct FastIcaParams<F: Float>(FastIcaValidParams<F>);

impl<F: Float> Default for FastIcaParams<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float> FastIca<F> {
    pub fn params() -> FastIcaParams<F> {
        FastIcaParams::new()
    }
}

impl<F: Float> FastIcaParams<F> {
    /// Create new FastICA algorithm with default values for its parameters
    pub fn new() -> Self {
        Self(FastIcaValidParams {
            ncomponents: None,
            gfunc: GFunc::Logcosh(1.),
            max_iter: 200,
            tol: F::cast(1e-4),
            random_state: None,
        })
    }

    /// Set the number of components to use, if not set all are used
    pub fn ncomponents(mut self, ncomponents: usize) -> Self {
        self.0.ncomponents = Some(ncomponents);
        self
    }

    /// G function used in the approximation to neg-entropy, refer [`GFunc`]
    pub fn gfunc(mut self, gfunc: GFunc) -> Self {
        self.0.gfunc = gfunc;
        self
    }

    /// Set maximum number of iterations during fit
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.0.max_iter = max_iter;
        self
    }

    /// Set tolerance on upate at each iteration
    pub fn tol(mut self, tol: F) -> Self {
        self.0.tol = tol;
        self
    }

    /// Set seed for random number generator for reproducible results.
    pub fn random_state(mut self, random_state: usize) -> Self {
        self.0.random_state = Some(random_state);
        self
    }
}

impl<F: Float> ParamGuard for FastIcaParams<F> {
    type Checked = FastIcaValidParams<F>;
    type Error = FastIcaError;

    fn check_ref(&self) -> Result<&Self::Checked, Self::Error> {
        if self.0.tol < F::zero() {
            Err(FastIcaError::InvalidTolerance(self.0.tol.to_f32().unwrap()))
        } else {
            Ok(&self.0)
        }
    }

    fn check(self) -> Result<Self::Checked, Self::Error> {
        self.check_ref()?;
        Ok(self.0)
    }
}
