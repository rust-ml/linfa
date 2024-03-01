use std::fmt::Debug;

use linfa::ParamGuard;
use rand::Rng;

use crate::ReductionError;

/// Sparse random projection hyperparameters
///
/// The main hyperparameter of a sparse random projection is
/// the dimension of the embedding.
/// This dimension is usually determined by the desired precision (or distortion) `eps`,
/// using the [Johnson-Lindenstrauss Lemma](https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma).
/// However, this lemma makes a very conservative estimate of the required dimension,
/// and does not leverage the structure of the data, therefore it is also possible
/// to manually specify the dimension of the embedding.
pub struct SparseRandomProjectionParams<R: Rng + Clone>(
    pub(crate) SparseRandomProjectionValidParams<R>,
);

impl<R: Rng + Clone> SparseRandomProjectionParams<R> {
    /// Set the dimension of output of the embedding.
    ///
    /// Setting the target dimension with this function
    /// discards the precision parameter if it had been set previously.
    pub fn target_dim(mut self, dim: usize) -> Self {
        self.0.params = SparseRandomProjectionParamsInner::Dimension { target_dim: dim };

        self
    }

    /// Set the precision (distortion, `eps`) of the embedding.
    ///
    /// Setting the precision with this function
    /// discards the target dimension parameter if it had been set previously.
    pub fn precision(mut self, eps: f64) -> Self {
        self.0.params = SparseRandomProjectionParamsInner::Precision { precision: eps };

        self
    }

    /// Specify the random number generator to use to generate the projection matrix.
    pub fn with_rng<R2: Rng + Clone>(self, rng: R2) -> SparseRandomProjectionParams<R2> {
        SparseRandomProjectionParams(SparseRandomProjectionValidParams {
            params: self.0.params,
            rng,
        })
    }
}

/// Sparse random projection hyperparameters
///
/// The main hyperparameter of a sparse random projection is
/// the dimension of the embedding.
/// This dimension is usually determined by the desired precision (or distortion) `eps`,
/// using the [Johnson-Lindenstrauss Lemma](https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma).
/// However, this lemma makes a very conservative estimate of the required dimension,
/// and does not leverage the structure of the data, therefore it is also possible
/// to manually specify the dimension of the embedding.
#[derive(Debug, Clone, PartialEq)]
pub struct SparseRandomProjectionValidParams<R> {
    pub(super) params: SparseRandomProjectionParamsInner,
    pub(super) rng: R,
}

/// Internal data structure that either holds the dimension or the embedding,
/// or the precision, which can be used later to compute the dimension
/// (see [super::super::common::johnson_lindenstrauss_min_dim]).
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum SparseRandomProjectionParamsInner {
    Dimension { target_dim: usize },
    Precision { precision: f64 },
}

impl SparseRandomProjectionParamsInner {
    fn target_dim(&self) -> Option<usize> {
        use SparseRandomProjectionParamsInner::*;
        match self {
            Dimension { target_dim } => Some(*target_dim),
            Precision { .. } => None,
        }
    }

    fn eps(&self) -> Option<f64> {
        use SparseRandomProjectionParamsInner::*;
        match self {
            Dimension { .. } => None,
            Precision { precision } => Some(*precision),
        }
    }
}

impl<R: Rng + Clone> SparseRandomProjectionValidParams<R> {
    pub fn target_dim(&self) -> Option<usize> {
        self.params.target_dim()
    }

    pub fn precision(&self) -> Option<f64> {
        self.params.eps()
    }

    pub fn rng(&self) -> &R {
        &self.rng
    }
}

impl<R: Rng + Clone> ParamGuard for SparseRandomProjectionParams<R> {
    type Checked = SparseRandomProjectionValidParams<R>;
    type Error = ReductionError;

    fn check_ref(&self) -> Result<&Self::Checked, Self::Error> {
        match self.0.params {
            SparseRandomProjectionParamsInner::Dimension { target_dim } => {
                if target_dim == 0 {
                    return Err(ReductionError::NonPositiveEmbeddingSize);
                }
            }
            SparseRandomProjectionParamsInner::Precision { precision } => {
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
