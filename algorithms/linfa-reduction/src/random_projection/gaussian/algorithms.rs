use linfa::{prelude::Records, traits::Fit, Float};
use ndarray::{Array, Array2, Ix2};
use ndarray_rand::{
    rand_distr::{Normal, StandardNormal},
    RandomExt,
};
use rand::{prelude::Distribution, rngs::SmallRng, Rng};

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

        let n_dims = match &self.params {
            GaussianRandomProjectionParamsInner::Dimension { target_dim } => *target_dim,
            GaussianRandomProjectionParamsInner::Precision { precision } => {
                johnson_lindenstrauss_min_dim(n_samples, *precision)
            }
        };

        let std_dev = F::cast(n_features).sqrt().recip();
        let gaussian = Normal::new(F::zero(), std_dev)?;

        let proj = match self.rng.clone() {
            Some(mut rng) => Array::random_using((n_features, n_dims), gaussian, &mut rng),
            None => Array::random((n_features, n_dims), gaussian),
        };

        Ok(GaussianRandomProjection { projection: proj })
    }
}

impl<F: Float> GaussianRandomProjection<F> {
    /// Create new parameters for a [`GaussianRandomProjection`] with default values
    /// `precision = 0.1` and no custom [`Rng`] provided.
    pub fn params() -> GaussianRandomProjectionParams<SmallRng> {
        GaussianRandomProjectionParams(GaussianRandomProjectionValidParams {
            params: GaussianRandomProjectionParamsInner::Precision { precision: 0.1 },
            rng: None,
        })
    }
}

impl_proj! {GaussianRandomProjection<F>}
