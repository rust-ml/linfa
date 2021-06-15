use linfa::Float;
#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};

#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Clone, Debug, PartialEq)]
/// The set of hyperparameters that can be specified for the execution of
/// the [Approximated DBSCAN algorithm](struct.AppxDbscan.html).
pub struct AppxDbscanHyperParams<F: Float> {
    /// Distance between points for them to be considered neighbours.
    pub(crate) tolerance: F,
    /// Minimum number of neighboring points a point needs to have to be a core
    /// point and not a noise point.
    pub(crate) min_points: usize,
    /// Approximation factor, allows the distance between two points
    /// for them to be considered neighbours to reach
    /// `tolerance * (1 + slack)`
    pub(crate) slack: F,
}

impl<F: Float> AppxDbscanHyperParams<F> {
    pub(crate) fn new(min_points: usize) -> Self {
        if min_points <= 1 {
            panic!("`min_points` must be greater than 1!");
        }

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
        if tolerance <= F::zero() {
            panic!("`tolerance` must be greater than 0!");
        }

        self.tolerance = tolerance;
        self
    }

    /// Set the slack
    pub fn slack(mut self, slack: F) -> Self {
        if slack <= F::zero() {
            panic!("`slack` must be greater than 0!");
        }

        self.slack = slack;
        self
    }

    /// Get the tolerance
    pub fn get_tolerance(&self) -> F {
        self.tolerance
    }

    /// Get the minimum number of points in a cluster
    pub fn get_minimum_points(&self) -> usize {
        self.min_points
    }

    /// Get the slack
    pub fn get_slack(&self) -> F {
        self.slack
    }

    /// Get the approximate tolerance (`tolerance * (1 + slack)`)
    pub fn get_appx_tolerance(&self) -> F {
        self.tolerance * (F::one() + self.slack)
    }
}
