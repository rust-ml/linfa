#![doc = include_str!("../README.md")]

use ndarray::Array2;
use ndarray_rand::rand::Rng;
use ndarray_rand::rand_distr::Normal;

use linfa::{dataset::DatasetBase, traits::Transformer, Float, ParamGuard};

mod error;
mod hyperparams;

pub use error::{Result, TSneError};
pub use hyperparams::{TSneParams, TSneValidParams};

impl<F: Float, R: Rng + Clone> Transformer<Array2<F>, Result<Array2<F>>> for TSneValidParams<F, R> {
    fn transform(&self, mut data: Array2<F>) -> Result<Array2<F>> {
        let (nfeatures, nsamples) = (data.ncols(), data.nrows());

        // validate parameter-data constraints
        if self.embedding_size() > nfeatures {
            return Err(TSneError::EmbeddingSizeTooLarge);
        }

        if F::cast(nsamples - 1) < F::cast(3) * self.perplexity() {
            return Err(TSneError::PerplexityTooLarge);
        }

        // estimate number of preliminary iterations if not given
        let preliminary_iter = match self.preliminary_iter() {
            Some(x) => *x,
            None => usize::min(self.max_iter() / 2, 250),
        };

        let mut data = data.as_slice_mut().unwrap();

        let mut rng = self.rng().clone();
        let normal = Normal::new(0.0, 1e-4 * 10e-4).unwrap();

        let mut embedding: Vec<F> = (0..nsamples * self.embedding_size())
            .map(|_| rng.sample(&normal))
            .map(F::cast)
            .collect();

        bhtsne::run(
            &mut data,
            nsamples,
            nfeatures,
            &mut embedding,
            self.embedding_size(),
            self.perplexity(),
            self.approx_threshold(),
            true,
            self.max_iter() as u64,
            preliminary_iter as u64,
            preliminary_iter as u64,
        );

        Array2::from_shape_vec((nsamples, self.embedding_size()), embedding).map_err(|e| e.into())
    }
}

impl<F: Float, R: Rng + Clone> Transformer<Array2<F>, Result<Array2<F>>> for TSneParams<F, R> {
    fn transform(&self, x: Array2<F>) -> Result<Array2<F>> {
        self.check_ref()?.transform(x)
    }
}

impl<T, F: Float, R: Rng + Clone>
    Transformer<DatasetBase<Array2<F>, T>, Result<DatasetBase<Array2<F>, T>>>
    for TSneValidParams<F, R>
{
    fn transform(&self, ds: DatasetBase<Array2<F>, T>) -> Result<DatasetBase<Array2<F>, T>> {
        let DatasetBase {
            records,
            targets,
            weights,
            ..
        } = ds;

        self.transform(records)
            .map(|new_records| DatasetBase::new(new_records, targets).with_weights(weights))
    }
}

impl<T, F: Float, R: Rng + Clone>
    Transformer<DatasetBase<Array2<F>, T>, Result<DatasetBase<Array2<F>, T>>> for TSneParams<F, R>
{
    fn transform(&self, ds: DatasetBase<Array2<F>, T>) -> Result<DatasetBase<Array2<F>, T>> {
        self.check_ref()?.transform(ds)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{Array, Array1, Axis};
    use ndarray_rand::{rand_distr::Normal, RandomExt};
    use rand::{rngs::SmallRng, SeedableRng};

    use linfa::{dataset::Dataset, metrics::SilhouetteScore};

    #[test]
    fn autotraits() {
        fn has_autotraits<T: Send + Sync + Sized + Unpin>() {}
        has_autotraits::<TSneParams<f64, rand::distributions::Uniform<f64>>>();
        has_autotraits::<TSneValidParams<f64, rand::distributions::Uniform<f64>>>();
        has_autotraits::<TSneError>();
    }

    #[test]
    fn iris_separate() -> Result<()> {
        let ds = linfa_datasets::iris();
        let rng = SmallRng::seed_from_u64(42);

        let ds = TSneParams::embedding_size_with_rng(2, rng)
            .perplexity(10.0)
            .approx_threshold(0.0)
            .transform(ds)?;

        assert!(ds.silhouette_score()? > 0.6);

        Ok(())
    }

    #[test]
    fn blob_separate() -> Result<()> {
        let mut rng = SmallRng::seed_from_u64(42);
        let entries: Array2<f64> = ndarray::concatenate(
            Axis(0),
            &[
                Array::random_using((100, 2), Normal::new(-10., 0.5).unwrap(), &mut rng).view(),
                Array::random_using((100, 2), Normal::new(10., 0.5).unwrap(), &mut rng).view(),
            ],
        )?;

        let targets = (0..200).map(|x| x < 100).collect::<Array1<_>>();
        let dataset = Dataset::new(entries, targets);

        let ds = TSneParams::embedding_size_with_rng(2, rng)
            .perplexity(60.0)
            .approx_threshold(0.0)
            .transform(dataset)?;

        assert_abs_diff_eq!(ds.silhouette_score()?, 0.945, epsilon = 0.01);

        Ok(())
    }

    #[test]
    #[should_panic(expected = "NegativePerplexity")]
    fn perplexity_panic() {
        let ds = linfa_datasets::iris();

        TSneParams::embedding_size(2)
            .perplexity(-10.0)
            .transform(ds)
            .unwrap();
    }

    #[test]
    #[should_panic(expected = "NegativeApproximationThreshold")]
    fn approx_threshold_panic() {
        let ds = linfa_datasets::iris();

        TSneParams::embedding_size(2)
            .approx_threshold(-10.0)
            .transform(ds)
            .unwrap();
    }
    #[test]
    #[should_panic(expected = "EmbeddingSizeTooLarge")]
    fn embedding_size_panic() {
        let ds = linfa_datasets::iris();

        TSneParams::embedding_size(5).transform(ds).unwrap();
    }
}
