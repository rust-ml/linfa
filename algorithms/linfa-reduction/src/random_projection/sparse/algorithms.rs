use linfa::{prelude::Records, traits::Fit, Float};
use ndarray::Ix2;
use ndarray_rand::rand_distr::StandardNormal;
use rand::{distributions::Bernoulli, prelude::Distribution, thread_rng, Rng};
use sprs::{CsMat, TriMat};

use super::super::common::johnson_lindenstrauss_min_dim;
use super::hyperparams::SparseRandomProjectionParamsInner;
use super::{SparseRandomProjectionParams, SparseRandomProjectionValidParams};
use crate::{impl_proj, ReductionError};

/// Embedding via sparse random projection
pub struct SparseRandomProjection<F: Float> {
    projection: CsMat<F>,
}

impl<F, Rec, T> Fit<Rec, T, ReductionError> for SparseRandomProjectionValidParams
where
    F: Float,
    Rec: Records<Elem = F>,
    StandardNormal: Distribution<F>,
{
    type Object = SparseRandomProjection<F>;

    fn fit(&self, dataset: &linfa::DatasetBase<Rec, T>) -> Result<Self::Object, ReductionError> {
        let n_samples = dataset.nsamples();
        let n_features = dataset.nfeatures();

        let n_dims = match &self.params {
            SparseRandomProjectionParamsInner::Dimension { target_dim } => *target_dim,
            SparseRandomProjectionParamsInner::Precision { precision } => {
                johnson_lindenstrauss_min_dim(n_samples, *precision)
            }
        };

        let scale = (n_features as f64).sqrt();
        let p = 1f64 / scale;
        let dist = SparseDistribution::new(F::cast(scale), p);
        let mut rng = thread_rng();

        let (mut row_inds, mut col_inds, mut values) = (Vec::new(), Vec::new(), Vec::new());
        for row in 0..n_features {
            for col in 0..n_dims {
                if let Some(x) = dist.sample(&mut rng) {
                    row_inds.push(row);
                    col_inds.push(col);
                    values.push(x);
                }
            }
        }

        // `proj` will be used as the RHS of a matrix multiplication in [`SparseRandomProjection::transform`],
        // therefore we convert it to `csc` storage.
        let proj = TriMat::from_triplets((n_features, n_dims), row_inds, col_inds, values).to_csc();

        Ok(SparseRandomProjection { projection: proj })
    }
}

/// Random variable that has value `Some(+/- scale)` with probability `p/2` each,
/// and [`None`] with probability `1-p`.
struct SparseDistribution<F: Float> {
    scale: F,
    b: Bernoulli,
}

impl<F: Float> SparseDistribution<F> {
    pub fn new(scale: F, p: f64) -> Self {
        SparseDistribution {
            scale,
            b: Bernoulli::new(p).expect("Parmeter `p` must be between 0 and 1."),
        }
    }
}

impl<F: Float> Distribution<Option<F>> for SparseDistribution<F> {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Option<F> {
        let non_zero = self.b.sample(rng);
        if non_zero {
            if rng.gen::<bool>() {
                Some(self.scale)
            } else {
                Some(-self.scale)
            }
        } else {
            None
        }
    }
}

impl<F: Float> SparseRandomProjection<F> {
    /// Create new parameters for a [`SparseRandomProjection`] with default value
    /// `precision = 0.1`.
    pub fn params() -> SparseRandomProjectionParams {
        SparseRandomProjectionParams(SparseRandomProjectionValidParams {
            params: SparseRandomProjectionParamsInner::Precision { precision: 0.1 },
        })
    }
}

impl_proj! {SparseRandomProjection<F>}
