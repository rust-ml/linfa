use linfa::Float;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

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
    tolerance: F,
    /// Minimum number of neighboring points a point needs to have to be a core
    /// point and not a noise point.
    min_points: usize,
    /// Approximation factor, allows the distance between two points
    /// for them to be considered neighbours to reach
    /// `tolerance * (1 + slack)`
    slack: F,
    appx_tolerance: F,
}

/// Helper struct used to construct a set of hyperparameters for
pub struct AppxDbscanHyperParamsBuilder<F: Float> {
    tolerance: F,
    min_points: usize,
    slack: F,
}

impl<F: Float> AppxDbscanHyperParamsBuilder<F> {
    /// Distance between points for them to be considered neighbours.
    pub fn tolerance(mut self, tolerance: F) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Approximation factor, allows the distance between two points
    /// for them to be considered neighbours to reach
    /// `tolerance * (1 + slack)`
    pub fn slack(mut self, slack: F) -> Self {
        self.slack = slack;
        self
    }

    /// Return an instance of `AppxDbscanHyperParams` after having performed
    /// validation checks on all hyperparameters.
    ///
    /// **Panics** if any of the validation checks fail.
    pub fn build(self) -> AppxDbscanHyperParams<F> {
        AppxDbscanHyperParams::build(self.tolerance, self.min_points, self.slack)
    }
}

impl<F: Float> AppxDbscanHyperParams<F> {
    /// Minimum number of neighboring points a point needs to have to be a core
    /// point and not a noise point.
    ///
    /// Defaults are provided if the optional parameters are not specified:
    /// * `tolerance = 1e-4`
    /// * `slack = 1e-2`
    // Violates the convention that new should return a value of type `Self`
    #[allow(clippy::new_ret_no_self)]
    pub fn new(min_points: usize) -> AppxDbscanHyperParamsBuilder<F> {
        let default_slack = F::from(1e-2).unwrap();
        let default_tolerance = F::from(1e-4).unwrap();

        AppxDbscanHyperParamsBuilder {
            min_points,
            tolerance: default_tolerance,
            slack: default_slack,
        }
    }

    /// Two points are considered neighbors if the euclidean distance between
    /// them is below the tolerance
    pub fn tolerance(&self) -> F {
        self.tolerance
    }

    /// Minimum number of a points in a neighborhood around a point for it to
    /// not be considered noise
    pub fn minimum_points(&self) -> usize {
        self.min_points
    }

    /// Approximation factor, allows the distance between two points
    /// for them to be considered neighbours to reach
    /// `tolerance * (1 + slack)`
    pub fn slack(&self) -> F {
        self.slack
    }

    pub fn appx_tolerance(&self) -> F {
        self.appx_tolerance
    }

    fn build(tolerance: F, min_points: usize, slack: F) -> Self {
        if tolerance <= F::from(0.).unwrap() {
            panic!("`tolerance` must be greater than 0!");
        }
        // There is always at least one neighbor to a point (itself)
        if min_points <= 1 {
            panic!("`min_points` must be greater than 1!");
        }

        if slack <= F::from(0.).unwrap() {
            panic!("`slack` must be greater than 0!");
        }
        Self {
            tolerance: tolerance,
            min_points: min_points,
            slack: slack,
            appx_tolerance: tolerance * (F::one() + slack),
        }
    }
}
