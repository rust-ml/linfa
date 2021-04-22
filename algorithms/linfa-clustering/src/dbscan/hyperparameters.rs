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
/// the [DBSCAN algorithm](struct.Dbscan.html).
pub struct DbscanHyperParams<F: Float> {
    /// Distance between points for them to be considered neighbours.
    tolerance: F,
    /// Minimum number of neighboring points a point needs to have to be a core
    /// point and not a noise point.
    min_points: usize,
}

/// Helper struct used to construct a set of hyperparameters for [DBSCAN algorithm](struct.Dbscan.html).
pub struct DbscanHyperParamsBuilder<F: Float> {
    tolerance: F,
    min_points: usize,
}

impl<F: Float> DbscanHyperParamsBuilder<F> {
    /// Distance between points for them to be considered neighbours.
    pub fn tolerance(mut self, tolerance: F) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Return an instance of `DbscanHyperParams` after having performed
    /// validation checks on all hyperparameters.
    ///
    /// **Panics** if any of the validation checks fail.
    pub fn build(&self) -> DbscanHyperParams<F> {
        DbscanHyperParams::build(self.tolerance, self.min_points)
    }
}

impl<F: Float> DbscanHyperParams<F> {
    /// Minimum number of neighboring points a point needs to have to be a core
    /// point and not a noise point.
    ///
    /// Defaults are provided if the optional parameters are not specified:
    /// * `tolerance = 1e-4`
    // Violates the convention that new should return a value of type `Self`
    #[allow(clippy::new_ret_no_self)]
    pub fn new(min_points: usize) -> DbscanHyperParamsBuilder<F> {
        DbscanHyperParamsBuilder {
            min_points,
            tolerance: F::cast(1e-4),
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

    fn build(tolerance: F, min_points: usize) -> Self {
        if tolerance <= F::zero() {
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
