use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
/// The set of hyperparameters that can be specified for the execution of
/// the [DBSCAN algorithm](struct.Dbscan.html).
pub struct DbscanHyperParams {
    /// Distance between points for them to be considered neighbours.
    tolerance: f64,
    /// Minimum number of neighboring points a point needs to have to be a core
    /// point and not a noise point.
    min_points: usize,
}

/// Helper struct used to construct a set of hyperparameters for
pub struct DbscanHyperParamsBuilder {
    tolerance: f64,
    min_points: usize,
}

impl DbscanHyperParamsBuilder {
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    pub fn build(self) -> DbscanHyperParams {
        DbscanHyperParams::build(self.tolerance, self.min_points)
    }
}

impl DbscanHyperParams {
    pub fn new(min_points: usize) -> DbscanHyperParamsBuilder {
        DbscanHyperParamsBuilder {
            min_points,
            tolerance: 1e-4,
        }
    }

    pub fn tolerance(&self) -> f64 {
        self.tolerance
    }

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
