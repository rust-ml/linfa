use linfa::Float;
use linfa_nn::{distance::Distance, NearestNeighbour};
#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};
use thiserror::Error;

#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Debug)]
/// The set of hyperparameters that can be specified for the execution of
/// the [DBSCAN algorithm](struct.Dbscan.html).
pub struct DbscanHyperParams<F: Float, D: Distance<F>, N: NearestNeighbour> {
    pub(crate) tolerance: F,
    pub(crate) min_points: usize,
    pub(crate) dist_fn: D,
    pub(crate) nn_algo: N,
}

#[derive(Debug)]
/// Helper struct for building a set of [DBSCAN hyperparameters](struct.DbscanHyperParams.html)
pub struct DbscanHyperParamsBuilder<F: Float, D: Distance<F>, N: NearestNeighbour> {
    tolerance: F,
    min_points: usize,
    dist_fn: D,
    nn_algo: N,
}

#[derive(Error, Debug)]
pub enum DbscanParamsError {
    #[error("min_points must be greater than 1")]
    MinPoints,
    #[error("tolerance must be greater than 0")]
    Tolerance,
}

impl<F: Float, D: Distance<F>, N: NearestNeighbour> DbscanHyperParamsBuilder<F, D, N> {
    pub(crate) fn new(min_points: usize, dist_fn: D, nn_algo: N) -> Self {
        DbscanHyperParamsBuilder {
            min_points,
            tolerance: F::cast(1e-4),
            dist_fn,
            nn_algo,
        }
    }

    /// Set the tolerance
    pub fn tolerance(mut self, tolerance: F) -> Self {
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

    /// Verify the values in the builder and return a set of hyperparameters
    pub fn build(self) -> Result<DbscanHyperParams<F, D, N>, DbscanParamsError> {
        if self.min_points <= 1 {
            Err(DbscanParamsError::MinPoints)
        } else if self.tolerance <= F::zero() {
            Err(DbscanParamsError::Tolerance)
        } else {
            Ok(DbscanHyperParams {
                min_points: self.min_points,
                tolerance: self.tolerance,
                nn_algo: self.nn_algo,
                dist_fn: self.dist_fn,
            })
        }
    }
}

impl<F: Float, D: Distance<F>, N: NearestNeighbour> DbscanHyperParams<F, D, N> {
    /// Nearest neighbour algorithm used for range queries
    pub fn tolerance(&self) -> F {
        self.tolerance
    }

    /// Minimum number of neighboring points a point needs to have to be a core                                                                                                
    /// point and not a noise point.
    pub fn minimum_points(&self) -> usize {
        self.min_points
    }

    /// Distance metric used in the DBSCAN calculation
    pub fn dist_fn(&self) -> &D {
        &self.dist_fn
    }

    /// Nearest neighbour algorithm used for range queries
    pub fn nn_algo(&self) -> &N {
        &self.nn_algo
    }
}

#[cfg(test)]
mod tests {
    use linfa_nn::{distance::L2Dist, CommonNearestNeighbour};

    use super::*;

    #[test]
    fn tolerance_cannot_be_zero() {
        assert!(
            DbscanHyperParamsBuilder::new(2, L2Dist, CommonNearestNeighbour::KdTree)
                .tolerance(0.0)
                .build()
                .is_err()
        );
    }

    #[test]
    fn min_points_at_least_2() {
        assert!(
            DbscanHyperParamsBuilder::new(1, L2Dist, CommonNearestNeighbour::KdTree)
                .tolerance(3.3)
                .build()
                .is_err()
        );
    }
}
