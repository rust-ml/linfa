use std::fmt::Debug;

use linfa::ParamGuard;
use rand::Rng;

use crate::ReductionError;

/// Gaussian random projection hyperparameters
///
/// The main hyperparameter of a gaussian random projection is
/// the dimension of the embedding.
/// This dimension is usually determined by the desired precision (or distortion) `eps`,
/// using the [Johnson-Lindenstrauss Lemma](https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma).
/// However, this lemma makes a very conservative estimate of the required dimension,
/// and does not leverage the structure of the data, therefore it is also possible
/// to manually specify the dimension of the embedding.
///
/// As this algorithm is randomized, it also accepts a [`Rng`] as parameter,
/// to be used to sample coordinate of the projection matrix.
pub struct GaussianRandomProjectionParams<R: Rng + Clone>(
    pub(crate) GaussianRandomProjectionValidParams<R>,
);

impl<R: Rng + Clone> GaussianRandomProjectionParams<R> {
    /// Set the dimension of output of the embedding.
    ///
    /// Setting the target dimension with this function
    /// discards the precision parameter if it had been set previously.
    pub fn target_dim(mut self, dim: usize) -> Self {
        self.0.params = GaussianRandomProjectionParamsInner::Dimension { target_dim: dim };

        self
    }

    /// Set the precision (distortion, `eps`) of the embedding.
    ///
    /// Setting the precision with this function
    /// discards the target dimension parameter if it had been set previously.
    pub fn precision(mut self, eps: f64) -> Self {
        self.0.params = GaussianRandomProjectionParamsInner::Precision { precision: eps };

        self
    }

    /// Specify the random number generator to use to generate the projection matrix.
    ///
    /// Optional: if no RNG is specified, uses the default RNG in [ndarray_rand::RandomExt].  
    pub fn with_rng<R2: Rng + Clone>(self, rng: R2) -> GaussianRandomProjectionParams<R2> {
        GaussianRandomProjectionParams(GaussianRandomProjectionValidParams {
            params: self.0.params,
            rng: Some(rng),
        })
    }
}

/// Gaussian random projection hyperparameters
///
/// The main hyperparameter of a gaussian random projection is
/// the dimension of the embedding.
/// This dimension is usually determined by the desired precision (or distortion) `eps`,
/// using the [Johnson-Lindenstrauss Lemma](https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma).
/// However, this lemma makes a very conservative estimate of the required dimension,
/// and does not leverage the structure of the data, therefore it is also possible
/// to manually specify the dimension of the embedding.
///
/// As this algorithm is randomized, it also accepts an [`Rng`] as optional parameter,
/// to be used to sample coordinate of the projection matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct GaussianRandomProjectionValidParams<R: Rng + Clone> {
    pub(super) params: GaussianRandomProjectionParamsInner,
    pub(super) rng: Option<R>,
}

/// Internal data structure that either holds the dimension or the embedding,
/// or the precision, which can be used later to compute the dimension
/// (see [super::super::common::johnson_lindenstrauss_min_dim]).
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum GaussianRandomProjectionParamsInner {
    Dimension { target_dim: usize },
    Precision { precision: f64 },
}

impl GaussianRandomProjectionParamsInner {
    fn target_dim(&self) -> Option<usize> {
        use GaussianRandomProjectionParamsInner::*;
        match self {
            Dimension { target_dim } => Some(*target_dim),
            Precision { .. } => None,
        }
    }

    fn eps(&self) -> Option<f64> {
        use GaussianRandomProjectionParamsInner::*;
        match self {
            Dimension { .. } => None,
            Precision { precision } => Some(*precision),
        }
    }
}

impl<R: Rng + Clone> GaussianRandomProjectionValidParams<R> {
    pub fn target_dim(&self) -> Option<usize> {
        self.params.target_dim()
    }

    pub fn precision(&self) -> Option<f64> {
        self.params.eps()
    }

    pub fn rng(&self) -> Option<&R> {
        self.rng.as_ref()
    }
}

impl<R: Rng + Clone> ParamGuard for GaussianRandomProjectionParams<R> {
    type Checked = GaussianRandomProjectionValidParams<R>;
    type Error = ReductionError;

    fn check_ref(&self) -> Result<&Self::Checked, Self::Error> {
        match self.0.params {
            GaussianRandomProjectionParamsInner::Dimension { target_dim } => {
                if target_dim == 0 {
                    return Err(ReductionError::NonPositiveEmbeddingSize);
                }
            }
            GaussianRandomProjectionParamsInner::Precision { precision } => {
                if precision <= 0. || precision >= 1. {
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
