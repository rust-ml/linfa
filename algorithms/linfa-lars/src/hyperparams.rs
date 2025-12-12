use linfa::{Float, ParamGuard};

use crate::error::LarsError;

#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct LarsValidParams<F> {
    /// Whether to calculate the intercept for this model. If set
    /// to false, no intercept will be used in calculations
    /// If true, the data will be centered before fitting.
    fit_intercept: bool,

    /// The maximum number of non-zero coefficients allowed in the solution.  
    /// Limits the number of selected features during the LARS path.
    n_nonzero_coefs: usize,

    /// The machine-precision regularization in the computation of the
    /// Cholesky diagonal factors. Increase this for very ill-conditioned
    /// systems.
    eps: F,

    /// Enables verbose output during fitting.
    verbose: bool,
}

pub struct LarsParams<F>(LarsValidParams<F>);

impl<F: Float> LarsValidParams<F> {
    pub fn fit_intercept(&self) -> bool {
        self.fit_intercept
    }

    pub fn n_nonzero_coefs(&self) -> usize {
        self.n_nonzero_coefs
    }

    pub fn verbose(&self) -> bool {
        self.verbose
    }

    pub fn eps(&self) -> F {
        self.eps
    }
}

impl<F: Float> Default for LarsParams<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float> LarsParams<F> {
    /// Create default Lars hyper parameters
    ///
    /// By default, an intercept will be fitted. To disable fitting an
    /// intercept, call `.with_intercept(false)` before calling `.fit()`.
    ///
    /// The feature matrix will not be normalized by default.
    pub fn new() -> Self {
        Self(LarsValidParams {
            fit_intercept: true,
            n_nonzero_coefs: 500,
            eps: F::epsilon(),
            verbose: false,
        })
    }

    /// Whether to calculate the intercept for this model.
    /// Defaults to `true` if not set.
    /// If set to false, no intercept will be used in calculations.
    pub fn fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.0.fit_intercept = fit_intercept;
        self
    }

    /// Set the target number of non-zero coefficients.
    pub fn n_nonzero_coefs(mut self, n_nonzero_coefs: usize) -> Self {
        self.0.n_nonzero_coefs = n_nonzero_coefs;
        self
    }

    /// Set the machine-precision regularization in the computation of the
    /// Cholesky diagonal factors. Increase this for very ill-conditioned
    /// systems. Unlike the ``tol`` parameter in some iterative
    /// optimization-based algorithms, this parameter does not control
    /// the tolerance of the optimization.
    pub fn eps(mut self, eps: F) -> Self {
        self.0.eps = eps;
        self
    }

    /// Set output verbosity.
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.0.verbose = verbose;
        self
    }
}

impl<F: Float> ParamGuard for LarsParams<F> {
    type Checked = LarsValidParams<F>;
    type Error = LarsError;

    /// Validate the hyper parameters
    fn check_ref(&self) -> Result<&Self::Checked, Self::Error> {
        if self.0.eps.is_negative() {
            Err(LarsError::InvalidEpsilon)
        } else {
            Ok(&self.0)
        }
    }

    fn check(self) -> Result<Self::Checked, Self::Error> {
        self.check_ref()?;
        Ok(self.0)
    }
}
