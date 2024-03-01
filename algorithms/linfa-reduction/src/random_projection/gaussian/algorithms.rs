use linfa::{prelude::Records, traits::Fit, Float};
use ndarray::{Array, Array2, Ix2};
use ndarray_rand::{
    rand_distr::{Normal, StandardNormal},
    RandomExt,
};
use rand::{prelude::Distribution, Rng, SeedableRng};
use rand_xoshiro::Xoshiro256Plus;

use super::super::common::johnson_lindenstrauss_min_dim;
use super::hyperparams::GaussianRandomProjectionParamsInner;
use super::{GaussianRandomProjectionParams, GaussianRandomProjectionValidParams};
use crate::{impl_proj, ReductionError};

/// Embedding via Gaussian random projection
pub struct GaussianRandomProjection<F: Float> {
    projection: Array2<F>,
}

impl<F, Rec, T, R> Fit<Rec, T, ReductionError> for GaussianRandomProjectionValidParams<R>
where
    F: Float,
    Rec: Records<Elem = F>,
    R: Rng + Clone,
    StandardNormal: Distribution<F>,
{
    type Object = GaussianRandomProjection<F>;

    fn fit(&self, dataset: &linfa::DatasetBase<Rec, T>) -> Result<Self::Object, ReductionError> {
        let n_samples = dataset.nsamples();
        let n_features = dataset.nfeatures();
        let mut rng = self.rng.clone();

        let n_dims = match &self.params {
            GaussianRandomProjectionParamsInner::Dimension { target_dim } => *target_dim,
            GaussianRandomProjectionParamsInner::Epsilon { eps } => {
                johnson_lindenstrauss_min_dim(n_samples, *eps)
            }
        };

        if n_dims > n_features {
            return Err(ReductionError::DimensionIncrease(n_dims, n_features));
        }

        let std_dev = F::cast(n_features).sqrt().recip();
        let gaussian = Normal::new(F::zero(), std_dev)?;

        let proj = Array::random_using((n_features, n_dims), gaussian, &mut rng);

        Ok(GaussianRandomProjection { projection: proj })
    }
}

impl<F: Float> GaussianRandomProjection<F> {
    /// Create new parameters for a [`GaussianRandomProjection`] with default value
    /// `eps = 0.1` and a [`Xoshiro256Plus`] RNG.
    pub fn params() -> GaussianRandomProjectionParams<Xoshiro256Plus> {
        GaussianRandomProjectionParams(GaussianRandomProjectionValidParams {
            params: GaussianRandomProjectionParamsInner::Epsilon { eps: 0.1 },
            rng: Xoshiro256Plus::seed_from_u64(42),
        })
    }

    /// Create new parameters for a [`GaussianRandomProjection`] with default values
    /// `eps = 0.1` and the provided [`Rng`].
    pub fn params_with_rng<R>(rng: R) -> GaussianRandomProjectionParams<R>
    where
        R: Rng + Clone,
    {
        GaussianRandomProjectionParams(GaussianRandomProjectionValidParams {
            params: GaussianRandomProjectionParamsInner::Epsilon { eps: 0.1 },
            rng,
        })
    }
}

impl_proj! {GaussianRandomProjection<F>}
