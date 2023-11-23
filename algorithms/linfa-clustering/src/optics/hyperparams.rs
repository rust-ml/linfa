use crate::optics::errors::{OpticsError, Result};
use linfa::{param_guard::TransformGuard, Float, ParamGuard};
#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
/// The set of hyperparameters that can be specified for the execution of
/// the [OPTICS algorithm](crate::Optics).
pub struct OpticsValidParams<F, D, N> {
    /// Distance between points for them to be considered neighbours.
    tolerance: F,
    /// Distance metric to be used for the algorithm
    dist_fn: D,
    /// Nearest Neighbour algorithm to use to find the nearest points
    nn_algo: N,
    /// Minimum number of neighboring points a point needs to have to be a core
    /// point and not a noise point.
    min_points: usize,
}

impl<F: Float, D, N> OpticsValidParams<F, D, N> {
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

    /// Distance metric to be used for the algorithm
    pub fn dist_fn(&self) -> &D {
        &self.dist_fn
    }

    /// Nearest Neighbour algorithm to use to find the nearest points
    pub fn nn_algo(&self) -> &N {
        &self.nn_algo
    }
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct OpticsParams<F, D, N>(OpticsValidParams<F, D, N>);

impl<F: Float, D, N> OpticsParams<F, D, N> {
    pub fn new(min_points: usize, dist_fn: D, nn_algo: N) -> Self {
        Self(OpticsValidParams {
            min_points,
            tolerance: F::infinity(),
            dist_fn,
            nn_algo,
        })
    }

    /// Distance between points for them to be considered neighbors. Compared to DBSCAN this
    /// parameter isn't strictly necessary but improves execution time by not considering every
    /// point. If the tolerance is too low the distances calculated are undefined and no clusters
    /// will be returned.
    pub fn tolerance(mut self, tolerance: F) -> Self {
        self.0.tolerance = tolerance;
        self
    }

    /// Distance metric to be used for the algorithm
    pub fn dist_fn(mut self, dist_fn: D) -> Self {
        self.0.dist_fn = dist_fn;
        self
    }

    /// Nearest Neighbour algorithm to use to find the nearest points
    pub fn nn_algo(mut self, nn_algo: N) -> Self {
        self.0.nn_algo = nn_algo;
        self
    }
}

impl<F: Float, D, N> ParamGuard for OpticsParams<F, D, N> {
    type Checked = OpticsValidParams<F, D, N>;
    type Error = OpticsError;

    fn check_ref(&self) -> Result<&Self::Checked> {
        if self.0.tolerance <= F::zero() {
            Err(OpticsError::InvalidValue(
                "`tolerance` must be greater than 0!".to_string(),
            ))
        } else if self.0.min_points <= 1 {
            // There is always at least one neighbor to a point (itself)
            Err(OpticsError::InvalidValue(
                "`min_points` must be greater than 1!".to_string(),
            ))
        } else {
            Ok(&self.0)
        }
    }

    fn check(self) -> Result<Self::Checked> {
        self.check_ref()?;
        Ok(self.0)
    }
}
impl<F: Float, D, N> TransformGuard for OpticsParams<F, D, N> {}
