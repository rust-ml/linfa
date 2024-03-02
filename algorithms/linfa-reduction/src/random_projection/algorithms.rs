use std::marker::PhantomData;

use linfa::{
    dataset::{AsTargets, FromTargetArray},
    prelude::Records,
    traits::{Fit, Transformer},
    DatasetBase, Float,
};
use ndarray::{linalg::Dot, Array2, ArrayBase, Data, Ix2};

use rand::{prelude::Distribution, Rng, SeedableRng};
use rand_xoshiro::Xoshiro256Plus;

use super::hyperparams::RandomProjectionParamsInner;
use super::{common::johnson_lindenstrauss_min_dim, methods::ProjectionMethod};
use super::{RandomProjectionParams, RandomProjectionValidParams};
use crate::ReductionError;

/// Embedding via random projection
pub struct RandomProjection<Proj: ProjectionMethod, F: Float>
where
    Proj::RandomDistribution: Distribution<F>,
{
    projection: Proj::ProjectionMatrix<F>,
}

impl<F, Proj, Rec, T, R> Fit<Rec, T, ReductionError> for RandomProjectionValidParams<Proj, R>
where
    F: Float,
    Proj: ProjectionMethod,
    Rec: Records<Elem = F>,
    R: Rng + Clone,
    Proj::RandomDistribution: Distribution<F>,
{
    type Object = RandomProjection<Proj, F>;

    fn fit(&self, dataset: &linfa::DatasetBase<Rec, T>) -> Result<Self::Object, ReductionError> {
        let n_samples = dataset.nsamples();
        let n_features = dataset.nfeatures();
        let mut rng = self.rng.clone();

        let n_dims = match &self.params {
            RandomProjectionParamsInner::Dimension { target_dim } => *target_dim,
            RandomProjectionParamsInner::Epsilon { eps } => {
                johnson_lindenstrauss_min_dim(n_samples, *eps)
            }
        };

        if n_dims > n_features {
            return Err(ReductionError::DimensionIncrease(n_dims, n_features));
        }

        let projection = Proj::generate_matrix(n_features, n_dims, &mut rng)?;

        Ok(RandomProjection { projection })
    }
}

impl<Proj: ProjectionMethod, F: Float> RandomProjection<Proj, F>
where
    Proj::RandomDistribution: Distribution<F>,
{
    /// Create new parameters for a [`RandomProjection`] with default value
    /// `eps = 0.1` and a [`Xoshiro256Plus`] RNG.
    pub fn params() -> RandomProjectionParams<Proj, Xoshiro256Plus> {
        RandomProjectionParams(RandomProjectionValidParams {
            params: RandomProjectionParamsInner::Epsilon { eps: 0.1 },
            rng: Xoshiro256Plus::seed_from_u64(42),
            marker: PhantomData,
        })
    }

    /// Create new parameters for a [`RandomProjection`] with default values
    /// `eps = 0.1` and the provided [`Rng`].
    pub fn params_with_rng<R>(rng: R) -> RandomProjectionParams<Proj, R>
    where
        R: Rng + Clone,
    {
        RandomProjectionParams(RandomProjectionValidParams {
            params: RandomProjectionParamsInner::Epsilon { eps: 0.1 },
            rng,
            marker: PhantomData,
        })
    }
}

impl<Proj, F, D> Transformer<&ArrayBase<D, Ix2>, Array2<F>> for RandomProjection<Proj, F>
where
    Proj: ProjectionMethod,
    F: Float,
    D: Data<Elem = F>,
    ArrayBase<D, Ix2>: Dot<Proj::ProjectionMatrix<F>, Output = Array2<F>>,
    Proj::RandomDistribution: Distribution<F>,
{
    /// Compute the embedding of a two-dimensional array
    fn transform(&self, x: &ArrayBase<D, Ix2>) -> Array2<F> {
        x.dot(&self.projection)
    }
}

impl<Proj, F, D> Transformer<ArrayBase<D, Ix2>, Array2<F>> for RandomProjection<Proj, F>
where
    Proj: ProjectionMethod,
    F: Float,
    D: Data<Elem = F>,
    ArrayBase<D, Ix2>: Dot<Proj::ProjectionMatrix<F>, Output = Array2<F>>,
    Proj::RandomDistribution: Distribution<F>,
{
    /// Compute the embedding of a two-dimensional array
    fn transform(&self, x: ArrayBase<D, Ix2>) -> Array2<F> {
        self.transform(&x)
    }
}

impl<Proj, F, T> Transformer<DatasetBase<Array2<F>, T>, DatasetBase<Array2<F>, T>>
    for RandomProjection<Proj, F>
where
    Proj: ProjectionMethod,
    F: Float,
    T: AsTargets,
    for<'a> ArrayBase<ndarray::ViewRepr<&'a F>, Ix2>:
        Dot<Proj::ProjectionMatrix<F>, Output = Array2<F>>,
    Proj::RandomDistribution: Distribution<F>,
{
    /// Compute the embedding of a dataset
    ///
    /// # Parameter
    ///
    /// * `data`: a dataset
    ///
    /// # Returns
    ///
    /// New dataset, with data equal to the embedding of the input data
    fn transform(&self, data: DatasetBase<Array2<F>, T>) -> DatasetBase<Array2<F>, T> {
        let new_records = self.transform(data.records().view());

        DatasetBase::new(new_records, data.targets)
    }
}

impl<'a, Proj, F, L, T> Transformer<&'a DatasetBase<Array2<F>, T>, DatasetBase<Array2<F>, T::View>>
    for RandomProjection<Proj, F>
where
    Proj: ProjectionMethod,
    F: Float,
    L: 'a,
    T: AsTargets<Elem = L> + FromTargetArray<'a>,
    for<'b> ArrayBase<ndarray::ViewRepr<&'b F>, Ix2>:
        Dot<Proj::ProjectionMatrix<F>, Output = Array2<F>>,
    Proj::RandomDistribution: Distribution<F>,
{
    /// Compute the embedding of a dataset
    ///
    /// # Parameter
    ///
    /// * `data`: a dataset
    ///
    /// # Returns
    ///
    /// New dataset, with data equal to the embedding of the input data
    fn transform(&self, data: &'a DatasetBase<Array2<F>, T>) -> DatasetBase<Array2<F>, T::View> {
        let new_records = self.transform(data.records().view());

        DatasetBase::new(
            new_records,
            T::new_targets_view(AsTargets::as_targets(data)),
        )
    }
}
