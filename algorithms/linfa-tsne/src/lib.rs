//! t-distributed stochastic neighbor embedding
//!
use ndarray::Array2;
use ndarray_rand::rand::{rngs::SmallRng, Rng, SeedableRng};
use ndarray_rand::rand_distr::Normal;

use linfa::{dataset::DatasetBase, traits::Transformer, Float};

mod error;
pub use error::{Result, TSneError};

/// The t-SNE algorithm is a statistical method for visualizing high-dimensional data by
/// giving each datapoint a location in a two or three-dimensional map.
///
/// The t-SNE algorithm comprises two main stages. First, t-SNE constructs a probability
/// distribution over pairs of high-dimensional objects in such a way that similar objects
/// are assigned a higher probability while dissimilar points are assigned a lower probability.
/// Second, t-SNE defines a similar probability distribution over the points in the low-dimensional
/// map, and it minimizes the Kullback–Leibler divergence (KL divergence) between the two
/// distributions with respect to the locations of the points in the map.
///
/// This crate wraps the [bhtsne](https://github.com/frjnn/bhtsne) crate for the linfa project. It
/// implements the exact t-SNE, as well as the Barnes-Hut approximation.
///
/// # Examples
///
/// ```
/// let ds = linfa_datasets::iris();
///
/// let ds = TSne::embedding_size(2)
///     .perplexity(10.0)
///     .approx_threshold(0.1)
///     .transform(ds)?;
/// ```
pub struct TSne<F: Float, R: Rng + Clone> {
    embedding_size: usize,
    threshold: F,
    perplexity: F,
    max_iter: usize,
    preliminary_iter: Option<usize>,
    rng: R,
}

impl<F: Float> TSne<F, SmallRng> {
    /// Create a t-SNE param set with given embedding size
    ///
    /// # Defaults to:
    ///  * `theta`: 0.5
    ///  * `perplexity`: 1.0
    ///  * `max_iter`: 2000
    ///  * `rng`: SmallRng with seed 42
    pub fn embedding_size(embedding_size: usize) -> TSne<F, SmallRng> {
        Self::embedding_size_with_rng(embedding_size, SmallRng::seed_from_u64(42))
    }
}

impl<F: Float, R: Rng + Clone> TSne<F, R> {
    /// Create a t-SNE param set with given embedding size and random number generator
    ///
    /// # Defaults to:
    ///  * `theta`: 0.5
    ///  * `perplexity`: 1.0
    ///  * `max_iter`: 2000
    pub fn embedding_size_with_rng(embedding_size: usize, rng: R) -> TSne<F, R> {
        TSne {
            embedding_size,
            rng,
            threshold: F::from(0.5).unwrap(),
            perplexity: F::from(5.0).unwrap(),
            max_iter: 2000,
            preliminary_iter: None,
        }
    }

    /// Set the approximation threshold of the Barnes Hut algorithm
    ///
    /// The threshold decides whether a cluster centroid can be used as a summary for the whole
    /// area. This was proposed by Barnes and Hut and compares the ratio of cell radius and
    /// distance to a factor theta. This threshold lies in range (0, inf) where a value of 0
    /// disables approximation and a positive value approximates the gradient with the cell center.
    pub fn approx_threshold(mut self, threshold: F) -> Self {
        self.threshold = threshold;

        self
    }

    /// Set the perplexity of the t-SNE algorithm
    pub fn perplexity(mut self, perplexity: F) -> Self {
        self.perplexity = perplexity;

        self
    }

    /// Set the maximal number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;

        self
    }

    /// Set the number of iterations after which the true P distribution is used
    ///
    /// At the beginning of the training process it is useful to multiply the P distribution values
    /// by a certain factor (here 12x) to get the global view right. After this number of iterations
    /// the true P distribution value is used. If None the number is estimated.
    pub fn preliminary_iter(mut self, num_iter: usize) -> Self {
        self.preliminary_iter = Some(num_iter);

        self
    }

    /// Validates parameters
    pub fn validate(&self, nfeatures: usize, nsamples: usize) -> Result<()> {
        if self.perplexity < F::zero() {
            return Err(TSneError::NegativePerplexity);
        }

        if self.threshold < F::zero() {
            return Err(TSneError::NegativeApproximationThreshold);
        }

        if self.embedding_size > nfeatures {
            return Err(TSneError::EmbeddingSizeTooLarge);
        }

        if F::from(nsamples - 1).unwrap() < F::from(3).unwrap() * self.perplexity {
            return Err(TSneError::PerplexityTooLarge);
        }

        Ok(())
    }
}

impl<F: Float, R: Rng + Clone> Transformer<Array2<F>, Result<Array2<F>>> for TSne<F, R> {
    fn transform(&self, mut data: Array2<F>) -> Result<Array2<F>> {
        let (nfeatures, nsamples) = (data.ncols(), data.nrows());
        self.validate(nfeatures, nsamples)?;

        let preliminary_iter = match self.preliminary_iter {
            Some(x) => x,
            None => usize::min(self.max_iter / 2, 250),
        };

        let mut data = data.as_slice_mut().unwrap();

        let mut rng = self.rng.clone();
        let normal = Normal::new(0.0, 1e-4 * 10e-4).unwrap();

        let mut embedding: Vec<F> = (0..nsamples * self.embedding_size)
            .map(|_| rng.sample(&normal))
            .map(|x| F::from(x).unwrap())
            .collect();

        bhtsne::run(
            &mut data,
            nsamples,
            nfeatures,
            &mut embedding,
            self.embedding_size,
            self.perplexity,
            self.threshold,
            true,
            self.max_iter as u64,
            preliminary_iter as u64,
            preliminary_iter as u64,
        );

        Array2::from_shape_vec((nsamples, self.embedding_size), embedding).map_err(|e| e.into())
    }
}

impl<T, F: Float, R: Rng + Clone>
    Transformer<DatasetBase<Array2<F>, T>, Result<DatasetBase<Array2<F>, T>>> for TSne<F, R>
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{Array, Array1, Axis};
    use ndarray_rand::{rand_distr::Normal, RandomExt};

    use linfa::{dataset::Dataset, metrics::SilhouetteScore};

    #[test]
    fn iris_separate() -> Result<()> {
        let ds = linfa_datasets::iris();
        let rng = SmallRng::seed_from_u64(42);

        let ds = TSne::embedding_size_with_rng(2, rng)
            .perplexity(10.0)
            .approx_threshold(0.1)
            .transform(ds)?;

        assert_abs_diff_eq!(ds.silhouette_score()?, 0.615, epsilon = 0.01);

        Ok(())
    }

    #[test]
    fn blob_separate() -> Result<()> {
        let mut rng = SmallRng::seed_from_u64(42);
        let entries: Array2<f64> = ndarray::stack(
            Axis(0),
            &[
                Array::random_using((100, 2), Normal::new(-10., 0.5).unwrap(), &mut rng).view(),
                Array::random_using((100, 2), Normal::new(10., 0.5).unwrap(), &mut rng).view(),
            ],
        )?;

        let targets = (0..200).map(|x| x < 100).collect::<Array1<_>>();
        let dataset = Dataset::new(entries.clone(), targets);

        let ds = TSne::embedding_size_with_rng(2, rng)
            .perplexity(60.0)
            .approx_threshold(0.5)
            .transform(dataset)?;

        assert_abs_diff_eq!(ds.silhouette_score()?, 0.927, epsilon = 0.01);

        Ok(())
    }

    #[test]
    #[should_panic(expected = "NegativePerplexity")]
    fn perplexity_panic() {
        let ds = linfa_datasets::iris();

        TSne::embedding_size(2)
            .perplexity(-10.0)
            .transform(ds)
            .unwrap();
    }

    #[test]
    #[should_panic(expected = "NegativeApproximationThreshold")]
    fn approx_threshold_panic() {
        let ds = linfa_datasets::iris();

        TSne::embedding_size(2)
            .approx_threshold(-10.0)
            .transform(ds)
            .unwrap();
    }
    #[test]
    #[should_panic(expected = "EmbeddingSizeTooLarge")]
    fn embedding_size_panic() {
        let ds = linfa_datasets::iris();

        TSne::embedding_size(5).transform(ds).unwrap();
    }
}
