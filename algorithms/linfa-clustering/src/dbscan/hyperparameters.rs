use linfa::Float;
use linfa_nn::{distance::Distance, NearestNeighbour};
#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};

#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Debug)]
/// The set of hyperparameters that can be specified for the execution of
/// the [DBSCAN algorithm](struct.Dbscan.html).
pub struct DbscanHyperParams<F: Float, D: Distance<F>, N: NearestNeighbour> {
    /// Distance between points for them to be considered neighbours.
    pub(crate) tolerance: F,
    /// Minimum number of neighboring points a point needs to have to be a core
    /// point and not a noise point.
    pub(crate) min_points: usize,
    /// Distance metric used in the DBSCAN calculation
    pub(crate) dist_fn: D,
    /// Nearest neighbour algorithm used for range queries
    pub(crate) nn_algo: N,
}

impl<F: Float, D: Distance<F>, N: NearestNeighbour> DbscanHyperParams<F, D, N> {
    pub(crate) fn new(min_points: usize, dist_fn: D, nn_algo: N) -> Self {
        if min_points <= 1 {
            panic!("`min_points` must be greater than 1!");
        }

        DbscanHyperParams {
            min_points,
            tolerance: F::cast(1e-4),
            dist_fn,
            nn_algo,
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

    /// Set the nearest neighbour algorithm to be used
    pub fn nn_algo(mut self, nn_algo: N) -> Self {
        self.nn_algo = nn_algo;
        self
    }

    /// Set the distance metric
    pub fn dist_fn(mut self, dist_fn: D) -> Self {
        self.dist_fn = dist_fn;
        self
    }

    /// Get the tolerance
    pub fn get_tolerance(&self) -> F {
        self.tolerance
    }

    /// Get the minimum number of points
    pub fn get_minimum_points(&self) -> usize {
        self.min_points
    }

    /// Get the distance metric
    pub fn get_dist_fn(&self) -> &D {
        &self.dist_fn
    }

    pub fn get_nn_algo(&self) -> &N {
        &self.nn_algo
    }
}

#[cfg(test)]
mod tests {
    use linfa_nn::{distance::L2Dist, CommonNearestNeighbour};

    use super::*;

    #[test]
    #[should_panic]
    fn tolerance_cannot_be_zero() {
        DbscanHyperParams::new(2, L2Dist, CommonNearestNeighbour::KdTree).tolerance(0.0);
    }

    #[test]
    #[should_panic]
    fn min_points_at_least_2() {
        DbscanHyperParams::new(1, L2Dist, CommonNearestNeighbour::KdTree).tolerance(3.3);
    }
}
