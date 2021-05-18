use linfa::Float;
use linfa_nn::{
    distance::{Distance, L2Dist},
    KdTree, NearestNeighbour,
};
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
pub struct DbscanHyperParams<F: Float, D: Distance<F>> {
    /// Distance between points for them to be considered neighbours.
    tolerance: F,
    /// Minimum number of neighboring points a point needs to have to be a core
    /// point and not a noise point.
    min_points: usize,
    /// Distance metric used in the DBSCAN calculation
    dist_fn: D,
    /// Nearest neighbour algorithm used for range queries
    nn_impl: Box<dyn NearestNeighbour<F, D>>,
}

/// Helper struct used to construct a set of hyperparameters for [DBSCAN algorithm](struct.Dbscan.html).
pub struct DbscanHyperParamsBuilder<F: Float, D: Distance<F>> {
    tolerance: F,
    min_points: usize,
    dist_fn: D,
    nn_impl: Box<dyn NearestNeighbour<F, D>>,
}

impl<F: Float> DbscanHyperParamsBuilder<F, L2Dist> {
    /// Configures the hyperparameters with the minimum number of points required to form a cluster
    ///
    /// Defaults are provided if the optional parameters are not specified:
    /// * `tolerance = 1e-4`
    /// * `nn_impl = KdTree`
    /// * `dist_fn = L2Dist` (Euclidean distance)
    // Violates the convention that new should return a value of type `Self`
    #[allow(clippy::new_ret_no_self)]
    pub fn new(min_points: usize) -> Self {
        Self::with_dist_fn(min_points, L2Dist)
    }
}

impl<F: Float, D: 'static + Distance<F>> DbscanHyperParamsBuilder<F, D> {
    /// Configures the hyperparameters with the minimum number of points and a custom distance
    /// metric
    pub fn with_dist_fn(min_points: usize, dist_fn: D) -> Self {
        DbscanHyperParamsBuilder {
            min_points,
            tolerance: F::cast(1e-4),
            dist_fn,
            nn_impl: Box::new(KdTree::new()),
        }
    }

    /// Distance between points for them to be considered neighbours.
    pub fn tolerance(mut self, tolerance: F) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Nearest neighbour algorithm used for range queries
    pub fn nn_impl(mut self, nn_impl: impl 'static + NearestNeighbour<F, D>) -> Self {
        self.nn_impl = Box::new(nn_impl);
        self
    }

    /// Return an instance of `DbscanHyperParams` after having performed
    /// validation checks on all hyperparameters.
    ///
    /// **Panics** if any of the validation checks fail.
    pub fn build(self) -> DbscanHyperParams<F, D> {
        if self.tolerance <= F::zero() {
            panic!("`tolerance` must be greater than 0!");
        }
        // There is always at least one neighbor to a point (itself)
        if self.min_points <= 1 {
            panic!("`min_points` must be greater than 1!");
        }
        DbscanHyperParams {
            tolerance: self.tolerance,
            min_points: self.min_points,
            nn_impl: self.nn_impl,
            dist_fn: self.dist_fn,
        }
    }
}

impl<F: Float, D: Distance<F>> DbscanHyperParams<F, D> {
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

    /// Distance metric used in the DBSCAN calculation
    pub fn dist_fn(&self) -> &D {
        &self.dist_fn
    }

    /// Nearest neighbour algorithm used for range queries
    pub fn nn_impl(&self) -> &dyn NearestNeighbour<F, D> {
        &*self.nn_impl
    }
}
