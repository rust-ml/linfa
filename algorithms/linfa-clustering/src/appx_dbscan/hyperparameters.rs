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
pub struct AppxDbscanHyperParams<F: Float> {
    pub(crate) tolerance: F,
    pub(crate) min_points: usize,
    pub(crate) slack: F,
}

#[derive(Debug)]
/// Helper struct for building a set of [Approximated DBSCAN
/// hyperparameters](struct.AppxDbscanHyperParams.html)
pub struct AppxDbscanHyperParamsBuilder<F: Float> {
    tolerance: F,
    min_points: usize,
    slack: F,
}

#[derive(Debug, Error)]
pub enum AppxDbscanParamsError {
    #[error("tolerance must be greater than 0")]
    Tolerance,
    #[error("min_points must be greater than 1")]
    MinPoints,
    #[error("slack must be greater than 0")]
    Slack,
}

impl<F: Float> AppxDbscanHyperParamsBuilder<F> {
    pub(crate) fn new(min_points: usize) -> Self {
        let default_slack = F::cast(1e-2);
        let default_tolerance = F::cast(1e-4);

        Self {
            min_points,
            tolerance: default_tolerance,
            slack: default_slack,
        }
    }

    /// Set the tolerance
    pub fn tolerance(mut self, tolerance: F) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Set the slack
    pub fn slack(mut self, slack: F) -> Self {
        self.slack = slack;
        self
    }

    pub fn build(self) -> Result<AppxDbscanHyperParams<F>, AppxDbscanParamsError> {
        if self.min_points <= 1 {
            Err(AppxDbscanParamsError::MinPoints)
        } else if self.tolerance <= F::zero() {
            Err(AppxDbscanParamsError::Tolerance)
        } else if self.slack <= F::zero() {
            Err(AppxDbscanParamsError::Slack)
        } else {
            Ok(AppxDbscanHyperParams {
                min_points: self.min_points,
                tolerance: self.tolerance,
                slack: self.slack,
            })
        }
    }
}

impl<F: Float> AppxDbscanHyperParams<F> {
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
}
