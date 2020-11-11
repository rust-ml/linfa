#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
/// The set of hyperparameters that can be specified for the execution of
/// the [OPTICS algorithm](struct.Optics.html).
pub struct OpticsHyperParams {
    /// Distance between points for them to be considered neighbours.
    tolerance: f64,
    /// Minimum number of neighboring points a point needs to have to be a core
    /// point and not a noise point.
    min_points: usize,
}

/// Helper struct used to construct a set of hyperparameters for
pub struct OpticsHyperParamsBuilder {
    tolerance: f64,
    min_points: usize,
}

impl OpticsHyperParamsBuilder {
    /// Distance between points for them to be considered neighbors. Compared to DBSCAN this
    /// parameter isn't strictly necessary but improves execution time by not considering every
    /// point. If the tolerance is too low the distances calculated are undefined and no clusters
    /// will be returned.
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Return an instance of `OpticsHyperParams` after having performed
    /// validation checks on all hyperparameters.
    ///
    /// **Panics** if any of the validation checks fail.
    pub fn build(self) -> OpticsHyperParams {
        OpticsHyperParams::build(self.tolerance, self.min_points)
    }
}

impl OpticsHyperParams {
    /// Minimum number of neighboring points a point needs to have to be a core
    /// point and not a noise point.
    ///
    /// Defaults are provided if the optional parameters are not specified:
    /// * `tolerance = f64::MAX`
    // Violates the convention that new should return a value of type `Self`
    #[allow(clippy::new_ret_no_self)]
    pub fn new(min_points: usize) -> OpticsHyperParamsBuilder {
        OpticsHyperParamsBuilder {
            min_points,
            tolerance: f64::MAX,
        }
    }

    /// Two points are considered neighbors if the euclidean distance between
    /// them is below the tolerance
    pub fn tolerance(&self) -> f64 {
        self.tolerance
    }

    /// Minimum number of a points in a neighborhood around a point for it to
    /// not be considered noise
    pub fn minimum_points(&self) -> usize {
        self.min_points
    }

    fn build(tolerance: f64, min_points: usize) -> Self {
        if tolerance <= 0. {
            panic!("`tolerance` must be greater than 0!");
        }
        // There is always at least one neighbor to a point (itself)
        if min_points <= 1 {
            panic!("`min_points` must be greater than 1!");
        }
        Self {
            tolerance,
            min_points,
        }
    }
}
