use linfa::param_guard::TransformGuard;
use linfa::prelude::*;
use linfa::Float;

#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};
use thiserror::Error;

#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Clone, Debug, PartialEq)]
/// The set of hyperparameters that can be specified for the execution of
/// the [Approximated DBSCAN algorithm](struct.AppxDbscan.html).
pub struct AppxDbscanValidParams<F: Float, N> {
    pub(crate) tolerance: F,
    pub(crate) min_points: usize,
    pub(crate) slack: F,
    pub(crate) nn_algo: N,
}

#[derive(Debug, Clone, PartialEq)]
/// Helper struct for building a set of [Approximated DBSCAN
/// hyperparameters](struct.AppxDbscanParams.html)
pub struct AppxDbscanParams<F: Float, N>(AppxDbscanValidParams<F, N>);

#[derive(Debug, Error)]
pub enum AppxDbscanParamsError {
    #[error("tolerance must be greater than 0")]
    Tolerance,
    #[error("min_points must be greater than 1")]
    MinPoints,
    #[error("slack must be greater than 0")]
    Slack,
}

impl<F: Float, N> AppxDbscanParams<F, N> {
    pub(crate) fn new(min_points: usize, nn_algo: N) -> Self {
        let default_slack = F::cast(1e-6);
        let default_tolerance = F::cast(1e-4);

        Self(AppxDbscanValidParams {
            min_points,
            tolerance: default_tolerance,
            slack: default_slack,
            nn_algo,
        })
    }

    /// Set the tolerance
    pub fn tolerance(mut self, tolerance: F) -> Self {
        self.0.tolerance = tolerance;
        self
    }

    /// Set the slack
    pub fn slack(mut self, slack: F) -> Self {
        self.0.slack = slack;
        self
    }

    /// Set the nearest neighbour algorithm to be used
    pub fn nn_algo(mut self, nn_algo: N) -> Self {
        self.0.nn_algo = nn_algo;
        self
    }
}

impl<F: Float, N> ParamGuard for AppxDbscanParams<F, N> {
    type Checked = AppxDbscanValidParams<F, N>;
    type Error = AppxDbscanParamsError;

    fn check_ref(&self) -> Result<&Self::Checked, Self::Error> {
        if self.0.min_points <= 1 {
            Err(AppxDbscanParamsError::MinPoints)
        } else if self.0.tolerance <= F::zero() {
            Err(AppxDbscanParamsError::Tolerance)
        } else if self.0.slack <= F::zero() {
            Err(AppxDbscanParamsError::Slack)
        } else {
            Ok(&self.0)
        }
    }

    fn check(self) -> Result<Self::Checked, Self::Error> {
        self.check_ref()?;
        Ok(self.0)
    }
}
impl<F: Float, N> TransformGuard for AppxDbscanParams<F, N> {}

impl<F: Float, N> AppxDbscanValidParams<F, N> {
    /// Distance between points for them to be considered neighbours.
    pub fn tolerance(&self) -> F {
        self.tolerance
    }

    /// Distance between points for them to be considered neighbours.
    pub fn minimum_points(&self) -> usize {
        self.min_points
    }

    /// Minimum number of neighboring points a point needs to have to be a core
    /// point and not a noise point.
    pub fn slack(&self) -> F {
        self.slack
    }

    /// Get the approximate tolerance (`tolerance * (1 + slack)`)
    /// Approximation factor, allows the distance between two points
    /// for them to be considered neighbours to reach
    /// `tolerance * (1 + slack)`
    pub fn appx_tolerance(&self) -> F {
        self.tolerance * (F::one() + self.slack)
    }

    /// Nearest neighbour algorithm used for range queries
    pub fn nn_algo(&self) -> &N {
        &self.nn_algo
    }
}
