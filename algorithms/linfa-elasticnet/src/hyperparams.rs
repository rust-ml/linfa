#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};

use linfa::{Float, ParamGuard};

use crate::error::ElasticNetError;

use super::Result;

#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Clone, Debug, PartialEq)]
pub struct ElasticNetValidParamsBase<F, const MULTI_TASK: bool> {
    penalty: F,
    l1_ratio: F,
    with_intercept: bool,
    max_iterations: u32,
    tolerance: F,
}

/// A verified hyper-parameter set ready for the estimation of a ElasticNet regression model
///
/// See [`ElasticNetParams`](crate::ElasticNetParams) for more information.
pub type ElasticNetValidParams<F> = ElasticNetValidParamsBase<F, false>;

/// A verified hyper-parameter set ready for the estimation of a multi-task ElasticNet regression model
///
/// See [`MultiTaskElasticNetParams`](crate::MultiTaskElasticNetParams) for more information.
pub type MultiTaskElasticNetValidParams<F> = ElasticNetValidParamsBase<F, true>;

impl<F: Float, const MULTI_TASK: bool> ElasticNetValidParamsBase<F, MULTI_TASK> {
    pub fn penalty(&self) -> F {
        self.penalty
    }

    pub fn l1_ratio(&self) -> F {
        self.l1_ratio
    }

    pub fn with_intercept(&self) -> bool {
        self.with_intercept
    }

    pub fn max_iterations(&self) -> u32 {
        self.max_iterations
    }

    pub fn tolerance(&self) -> F {
        self.tolerance
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ElasticNetParamsBase<F, const MULTI_TASK: bool>(
    ElasticNetValidParamsBase<F, MULTI_TASK>,
);

/// A hyper-parameter set for Elastic-Net
///
/// Configures and minimizes the following objective function:
/// ```ignore
/// 1 / (2 * n_samples) * ||y - Xw||^2_2
///     + penalty * l1_ratio * ||w||_1
///     + 0.5 * penalty * (1 - l1_ratio) * ||w||^2_2
/// ```
///
/// The parameter set can be verified into a
/// [`ElasticNetValidParams`](crate::hyperparams::ElasticNetValidParams) by calling
/// [ParamGuard::check](Self::check). It is also possible to directly fit a model with
/// [Fit::fit](linfa::traits::Fit::fit) which implicitely verifies the parameter set prior to the
/// model estimation and forwards any error.
///
/// # Parameters
/// | Name | Default | Purpose | Range |
/// | :--- | :--- | :---| :--- |
/// | [penalty](Self::penalty) | `1.0` | Overall parameter penalty | `[0, inf)` |
/// | [l1_ratio](Self::l1_ratio) | `0.5` | Distribution of penalty to L1 and L2 regularizations | `[0.0, 1.0]` |
/// | [with_intercept](Self::with_intercept) | `true` | Enable intercept | `false`, `true` |
/// | [tolerance](Self::tolerance) | `1e-4` | Absolute change of any of the parameters | `(0, inf)` |
/// | [max_iterations](Self::max_iterations) | `1000` | Maximum number of iterations | `[1, inf)` |
///
/// # Errors
///
/// The following errors can come from invalid hyper-parameters:
///
/// Returns [`InvalidPenalty`](ElasticNetError::InvalidPenalty) if the penalty is negative.
///
/// Returns [`InvalidL1Ratio`](ElasticNetError::InvalidL1Ratio) if the L1 ratio is not in unit.
/// range
///
/// Returns [`InvalidTolerance`](ElasticNetError::InvalidTolerance) if the tolerance is negative.
///
/// # Example
///
/// ```rust
/// use linfa_elasticnet::{ElasticNetParams, ElasticNetError};
/// use linfa::prelude::*;
/// use ndarray::array;
///
/// let ds = Dataset::new(array![[1.0, 0.0], [0.0, 1.0]], array![3.0, 2.0]);
///
/// // create a new parameter set with penalty equals `1e-5`
/// let unchecked_params = ElasticNetParams::new()
///     .penalty(1e-5);
///
/// // fit model with unchecked parameter set
/// let model = unchecked_params.fit(&ds)?;
///
/// // transform into a verified parameter set
/// let checked_params = unchecked_params.check()?;
///
/// // Regenerate model with the verified parameters, this only returns
/// // errors originating from the fitting process
/// let model = checked_params.fit(&ds)?;
/// # Ok::<(), ElasticNetError>(())
/// ```
pub type ElasticNetParams<F> = ElasticNetParamsBase<F, false>;

/// A hyper-parameter set for multi-task Elastic-Net
///
/// The multi-task version (Y becomes a measurement matrix) is also supported and
/// solves the following objective function:
/// ```ignore
/// 1 / (2 * n_samples) * || Y - XW ||^2_F
///     + penalty * l1_ratio * ||W||_2,1
///     + 0.5 * penalty * (1 - l1_ratio) * ||W||^2_F
/// ```
///
/// See [`ElasticNetParams`](crate::ElasticNetParams) for information on parameters and return
/// values.
///
/// # Example
///
/// ```rust
/// use linfa_elasticnet::{MultiTaskElasticNetParams, ElasticNetError};
/// use linfa::prelude::*;
/// use ndarray::array;
///
/// let ds = Dataset::new(array![[1.0, 0.0], [0.0, 1.0]], array![[3.0, 1.1], [2.0, 2.2]]);
///
/// // create a new parameter set with penalty equals `1e-5`
/// let unchecked_params = MultiTaskElasticNetParams::new()
///     .penalty(1e-5);
///
/// // fit model with unchecked parameter set
/// let model = unchecked_params.fit(&ds)?;
///
/// // transform into a verified parameter set
/// let checked_params = unchecked_params.check()?;
///
/// // Regenerate model with the verified parameters, this only returns
/// // errors originating from the fitting process
/// let model = checked_params.fit(&ds)?;
/// # Ok::<(), ElasticNetError>(())
/// ```
pub type MultiTaskElasticNetParams<F> = ElasticNetParamsBase<F, true>;

impl<F: Float, const MULTI_TASK: bool> Default for ElasticNetParamsBase<F, MULTI_TASK> {
    fn default() -> Self {
        Self::new()
    }
}

/// Configure and fit a Elastic Net model
impl<F: Float, const MULTI_TASK: bool> ElasticNetParamsBase<F, MULTI_TASK> {
    /// Create default elastic net hyper parameters
    ///
    /// By default, an intercept will be fitted. To disable fitting an
    /// intercept, call `.with_intercept(false)` before calling `.fit()`.
    ///
    /// To additionally normalize the feature matrix before fitting, call
    /// `fit_intercept_and_normalize()` before calling `fit()`. The feature
    /// matrix will not be normalized by default.
    pub fn new() -> ElasticNetParamsBase<F, MULTI_TASK> {
        Self(ElasticNetValidParamsBase {
            penalty: F::one(),
            l1_ratio: F::cast(0.5),
            with_intercept: true,
            max_iterations: 1000,
            tolerance: F::cast(1e-4),
        })
    }

    /// Set the overall parameter penalty parameter of the elastic net, otherwise known as `alpha`.
    /// Use `l1_ratio` to configure how the penalty distributed to L1 and L2
    /// regularization.
    pub fn penalty(mut self, penalty: F) -> Self {
        self.0.penalty = penalty;
        self
    }

    /// Set l1_ratio parameter of the elastic net. Controls how the parameter
    /// penalty is distributed to L1 and L2 regularization.
    /// Setting `l1_ratio` to 1.0 is equivalent to a "Lasso" penalization,
    /// setting it to 0.0 is equivalent to "Ridge" penalization.
    ///
    /// Defaults to `0.5` if not set
    ///
    /// `l1_ratio` must be between `0.0` and `1.0`.
    pub fn l1_ratio(mut self, l1_ratio: F) -> Self {
        self.0.l1_ratio = l1_ratio;
        self
    }

    /// Configure the elastic net model to fit an intercept.
    /// Defaults to `true` if not set.
    pub fn with_intercept(mut self, with_intercept: bool) -> Self {
        self.0.with_intercept = with_intercept;
        self
    }

    /// Set the tolerance which is the minimum absolute change in any of the
    /// model parameters needed for the parameter optimization to continue.
    ///
    /// Defaults to `1e-4` if not set
    pub fn tolerance(mut self, tolerance: F) -> Self {
        self.0.tolerance = tolerance;
        self
    }

    /// Set the maximum number of iterations for the optimization routine.
    ///
    /// Defaults to `1000` if not set
    pub fn max_iterations(mut self, max_iterations: u32) -> Self {
        self.0.max_iterations = max_iterations;
        self
    }
}

impl<F: Float, const MULTI_TASK: bool> ParamGuard for ElasticNetParamsBase<F, MULTI_TASK> {
    type Checked = ElasticNetValidParamsBase<F, MULTI_TASK>;
    type Error = ElasticNetError;

    /// Validate the hyper parameters
    fn check_ref(&self) -> Result<&Self::Checked> {
        if self.0.penalty.is_negative() {
            Err(ElasticNetError::InvalidPenalty(
                self.0.penalty.to_f32().unwrap(),
            ))
        } else if !(F::zero()..=F::one()).contains(&self.0.l1_ratio) {
            Err(ElasticNetError::InvalidL1Ratio(
                self.0.l1_ratio.to_f32().unwrap(),
            ))
        } else if self.0.tolerance.is_negative() {
            Err(ElasticNetError::InvalidTolerance(
                self.0.tolerance.to_f32().unwrap(),
            ))
        } else {
            Ok(&self.0)
        }
    }

    fn check(self) -> Result<Self::Checked> {
        self.check_ref()?;
        Ok(self.0)
    }
}
