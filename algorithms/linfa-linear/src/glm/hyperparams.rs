use crate::{glm::link::Link, LinearError, TweedieRegressor};
use linfa::{Float, ParamGuard};
#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};

/// The set of hyperparameters that can be specified for the execution of the Tweedie Regressor.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct TweedieRegressorValidParams<F> {
    alpha: F,
    fit_intercept: bool,
    power: F,
    link: Option<Link>,
    max_iter: usize,
    tol: F,
}

impl<F: Float> TweedieRegressorValidParams<F> {
    pub fn alpha(&self) -> F {
        self.alpha
    }

    pub fn fit_intercept(&self) -> bool {
        self.fit_intercept
    }

    pub fn power(&self) -> F {
        self.power
    }

    pub fn link(&self) -> Link {
        match self.link {
            Some(x) => x,
            None if self.power <= F::zero() => Link::Identity,
            None => Link::Log,
        }
    }

    pub fn max_iter(&self) -> usize {
        self.max_iter
    }

    pub fn tol(&self) -> F {
        self.tol
    }
}

/// The set of hyperparameters that can be specified for the execution of the Tweedie Regressor.
#[derive(Debug, Clone, PartialEq)]
pub struct TweedieRegressorParams<F>(TweedieRegressorValidParams<F>);

impl<F: Float> Default for TweedieRegressorParams<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float> TweedieRegressor<F> {
    pub fn params() -> TweedieRegressorParams<F> {
        TweedieRegressorParams::new()
    }
}

impl<F: Float> TweedieRegressorParams<F> {
    pub fn new() -> Self {
        Self(TweedieRegressorValidParams {
            alpha: F::one(),
            fit_intercept: true,
            power: F::one(),
            link: None,
            max_iter: 100,
            tol: F::cast(1e-4),
        })
    }

    /// Constant that multiplies with the penalty term and thus determines the
    /// regularization strenght. `alpha` set to 0 is equivalent to unpenalized GLM.
    pub fn alpha(mut self, alpha: F) -> Self {
        self.0.alpha = alpha;
        self
    }

    /// Specifies whether a bias or intercept should be added to the model
    pub fn fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.0.fit_intercept = fit_intercept;
        self
    }

    /// The power determines the underlying target distribution
    pub fn power(mut self, power: F) -> Self {
        self.0.power = power;
        self
    }

    /// The link function of the GLM, for mapping from linear predictor `x @ coeff + intercept` to
    /// the prediction. If no value is set, the link will be selected based on the following,
    /// - [`identity`](Link::Identity) for Normal distribution (`power` = 0)
    /// - [`log`](Link::Log) for Poisson, Gamma and Inverse Gaussian distributions (`power` >= 1)
    pub fn link(mut self, link: Link) -> Self {
        self.0.link = Some(link);
        self
    }

    /// Maximum number of iterations for the LBFGS solver
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.0.max_iter = max_iter;
        self
    }

    /// Stopping criterion for the LBFGS solver
    pub fn tol(mut self, tol: F) -> Self {
        self.0.tol = tol;
        self
    }
}

impl<F: Float> ParamGuard for TweedieRegressorParams<F> {
    type Checked = TweedieRegressorValidParams<F>;
    type Error = LinearError<F>;

    fn check_ref(&self) -> Result<&Self::Checked, Self::Error> {
        if self.0.alpha.is_sign_negative() {
            Err(LinearError::InvalidPenalty(self.0.alpha))
        } else if self.0.power > F::zero() && self.0.power < F::one() {
            Err(LinearError::InvalidTweediePower(self.0.power))
        } else {
            Ok(&self.0)
        }
    }

    fn check(self) -> Result<Self::Checked, Self::Error> {
        self.check_ref()?;
        Ok(self.0)
    }
}
