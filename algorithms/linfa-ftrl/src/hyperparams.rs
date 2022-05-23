use crate::error::FtrlError;
use linfa::{Float, ParamGuard};
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialOrd, PartialEq)]
pub struct FtrlParams<F: Float, R: Rng>(pub(crate) FtrlValidParams<F, R>);

/// A verified hyper-parameter set ready for the estimation of a Follow the regularized leader - proximal model
///
/// See [`FtrlParams`](crate::FtrlParams) for more information.
#[derive(Clone, Debug, Serialize, Deserialize, PartialOrd, PartialEq)]
pub struct FtrlValidParams<F: Float, R: Rng> {
    pub(crate) alpha: F,
    pub(crate) beta: F,
    pub(crate) l1_ratio: F,
    pub(crate) l2_ratio: F,
    pub(crate) rng: R,
}

impl<F: Float, R: Rng> ParamGuard for FtrlParams<F, R> {
    type Checked = FtrlValidParams<F, R>;
    type Error = FtrlError;

    /// Validate the hyper parameters
    fn check_ref(&self) -> Result<&Self::Checked, Self::Error> {
        if !(F::zero()..=F::one()).contains(&self.0.l1_ratio) {
            Err(FtrlError::InvalidL1Ratio(self.0.l1_ratio.to_f32().unwrap()))
        } else if !(F::zero()..=F::one()).contains(&self.0.l2_ratio) {
            Err(FtrlError::InvalidL2Ratio(self.0.l2_ratio.to_f32().unwrap()))
        } else if !&self.0.alpha.is_finite() || self.0.alpha.is_negative() {
            Err(FtrlError::InvalidAlpha(self.0.alpha.to_f32().unwrap()))
        } else if !&self.0.beta.is_finite() || self.0.beta.is_negative() {
            Err(FtrlError::InvalidBeta(self.0.beta.to_f32().unwrap()))
        } else {
            Ok(&self.0)
        }
    }

    fn check(self) -> Result<Self::Checked, Self::Error> {
        self.check_ref()?;
        Ok(self.0)
    }
}

impl<F: Float, R: Rng> FtrlValidParams<F, R> {
    pub fn alpha(&self) -> F {
        self.alpha
    }

    pub fn beta(&self) -> F {
        self.beta
    }

    pub fn l1_ratio(&self) -> F {
        self.l1_ratio
    }

    pub fn l2_ratio(&self) -> F {
        self.l2_ratio
    }

    pub fn rng(&self) -> &R {
        &self.rng
    }
}

impl<F: Float, R: Rng> FtrlParams<F, R> {
    /// Create new hyperparameters with pre-defined values
    pub fn new(alpha: F, beta: F, l1_ratio: F, l2_ratio: F, rng: R) -> Self {
        Self(FtrlValidParams {
            alpha,
            beta,
            l1_ratio,
            l2_ratio,
            rng,
        })
    }

    /// Create new hyperparameters with pre-defined random number generator
    pub fn default_with_rng(rng: R) -> Self {
        Self(FtrlValidParams {
            alpha: F::cast(0.005),
            beta: F::cast(0.0),
            l1_ratio: F::cast(0.5),
            l2_ratio: F::cast(0.5),
            rng,
        })
    }

    /// Set the learning rate.
    ///
    /// Defaults to `0.005` if not set
    ///
    /// `alpha` must be positive and finite
    pub fn alpha(mut self, alpha: F) -> Self {
        self.0.alpha = alpha;
        self
    }

    /// Set the beta parameter.
    ///
    /// Defaults to `0.0` if not set
    ///
    /// `beta` must be positive and finite
    pub fn beta(mut self, beta: F) -> Self {
        self.0.beta = beta;
        self
    }

    /// Set l1_ratio parameter. Controls how the parameter
    ///
    /// Defaults to `0.5` if not set
    ///
    /// `l1_ratio` must be between `0.0` and `1.0`.
    pub fn l1_ratio(mut self, l1_ratio: F) -> Self {
        self.0.l1_ratio = l1_ratio;
        self
    }

    /// Set l2_ratio parameter. Controls how the parameter
    /// penalty is distributed to L2 regularization.
    ///
    /// Defaults to `0.5` if not set
    ///
    /// `l2_ratio` must be between `0.0` and `1.0`.
    pub fn l2_ratio(mut self, l2_ratio: F) -> Self {
        self.0.l2_ratio = l2_ratio;
        self
    }

    /// Set random number generator. Used to initialize Z values
    ///
    /// Defaults to Xoshiro256Plus
    ///
    /// `rng` must have Clone trait implemented.
    pub fn rng(mut self, rng: R) -> Self {
        self.0.rng = rng;
        self
    }
}
