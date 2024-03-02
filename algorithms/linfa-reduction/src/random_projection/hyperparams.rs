use std::{fmt::Debug, marker::PhantomData};

use linfa::ParamGuard;

use rand::Rng;

use crate::ReductionError;

use super::methods::ProjectionMethod;

/// Random projection hyperparameters
///
/// The main hyperparameter of random projections is
/// the dimension of the embedding.
/// This dimension is usually determined by the desired precision (or distortion) `eps`,
/// using the [Johnson-Lindenstrauss Lemma](https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma).
/// However, this lemma makes a very conservative estimate of the required dimension,
/// and does not leverage the structure of the data, therefore it is also possible
/// to manually specify the dimension of the embedding.
///
/// As this algorithm is randomized, it also accepts an [`Rng`] as parameter,
/// to be used to sample coordinate of the projection matrix.
pub struct RandomProjectionParams<Proj: ProjectionMethod, R: Rng + Clone>(
    pub(crate) RandomProjectionValidParams<Proj, R>,
);

impl<Proj: ProjectionMethod, R: Rng + Clone> RandomProjectionParams<Proj, R> {
    /// Set the dimension of output of the embedding.
    ///
    /// Setting the target dimension with this function
    /// discards the precision parameter if it had been set previously.
    pub fn target_dim(mut self, dim: usize) -> Self {
        self.0.params = RandomProjectionParamsInner::Dimension { target_dim: dim };

        self
    }

    /// Set the precision parameter (distortion, `eps`) of the embedding.
    ///
    /// Setting `eps` with this function
    /// discards the target dimension parameter if it had been set previously.
    pub fn eps(mut self, eps: f64) -> Self {
        self.0.params = RandomProjectionParamsInner::Epsilon { eps };

        self
    }

    /// Specify the random number generator to use to generate the projection matrix.
    pub fn with_rng<R2: Rng + Clone>(self, rng: R2) -> RandomProjectionParams<Proj, R2> {
        RandomProjectionParams(RandomProjectionValidParams {
            params: self.0.params,
            rng,
            marker: PhantomData,
        })
    }
}

/// Random projection hyperparameters
///
/// The main hyperparameter of random projections is
/// the dimension of the embedding.
/// This dimension is usually determined by the desired precision (or distortion) `eps`,
/// using the [Johnson-Lindenstrauss Lemma](https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma).
/// However, this lemma makes a very conservative estimate of the required dimension,
/// and does not leverage the structure of the data, therefore it is also possible
/// to manually specify the dimension of the embedding.
///
/// As this algorithm is randomized, it also accepts an [`Rng`] as parameter,
/// to be used to sample coordinate of the projection matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct RandomProjectionValidParams<Proj: ProjectionMethod, R: Rng + Clone> {
    pub(super) params: RandomProjectionParamsInner,
    pub(super) rng: R,
    pub(crate) marker: PhantomData<Proj>,
}

/// Internal data structure that either holds the dimension or the embedding,
/// or the precision, which can be used later to compute the dimension
/// (see [super::common::johnson_lindenstrauss_min_dim]).
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum RandomProjectionParamsInner {
    Dimension { target_dim: usize },
    Epsilon { eps: f64 },
}

impl RandomProjectionParamsInner {
    fn target_dim(&self) -> Option<usize> {
        use RandomProjectionParamsInner::*;
        match self {
            Dimension { target_dim } => Some(*target_dim),
            Epsilon { .. } => None,
        }
    }

    fn eps(&self) -> Option<f64> {
        use RandomProjectionParamsInner::*;
        match self {
            Dimension { .. } => None,
            Epsilon { eps } => Some(*eps),
        }
    }
}

impl<Proj: ProjectionMethod, R: Rng + Clone> RandomProjectionValidParams<Proj, R> {
    pub fn target_dim(&self) -> Option<usize> {
        self.params.target_dim()
    }

    pub fn eps(&self) -> Option<f64> {
        self.params.eps()
    }

    pub fn rng(&self) -> &R {
        &self.rng
    }
}

impl<Proj: ProjectionMethod, R: Rng + Clone> ParamGuard for RandomProjectionParams<Proj, R> {
    type Checked = RandomProjectionValidParams<Proj, R>;
    type Error = ReductionError;

    fn check_ref(&self) -> Result<&Self::Checked, Self::Error> {
        match self.0.params {
            RandomProjectionParamsInner::Dimension { target_dim } => {
                if target_dim == 0 {
                    return Err(ReductionError::NonPositiveEmbeddingSize);
                }
            }
            RandomProjectionParamsInner::Epsilon { eps } => {
                if eps <= 0. || eps >= 1. {
                    return Err(ReductionError::InvalidPrecision);
                }
            }
        };
        Ok(&self.0)
    }

    fn check(self) -> Result<Self::Checked, Self::Error> {
        self.check_ref()?;
        Ok(self.0)
    }
}
