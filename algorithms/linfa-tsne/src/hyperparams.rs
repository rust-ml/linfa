use linfa::{Float, ParamGuard};
use ndarray_rand::rand::{rngs::SmallRng, Rng, SeedableRng};

use crate::TSneError;

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
/// ```no_run
/// use linfa::traits::Transformer;
/// use linfa_tsne::TSneParams;
///
/// let ds = linfa_datasets::iris();
///
/// let ds = TSneParams::embedding_size(2)
///     .perplexity(10.0)
///     .approx_threshold(0.6)
///     .transform(ds);
/// ```

/// A verified hyper-parameter set ready for prediction
#[derive(Debug, Clone, PartialOrd, PartialEq)]
pub struct TSneValidParams<F, R> {
    embedding_size: usize,
    approx_threshold: F,
    perplexity: F,
    max_iter: usize,
    preliminary_iter: Option<usize>,
    rng: R,
}

impl<F: Float, R> TSneValidParams<F, R> {
    pub fn embedding_size(&self) -> usize {
        self.embedding_size
    }

    pub fn approx_threshold(&self) -> F {
        self.approx_threshold
    }

    pub fn perplexity(&self) -> F {
        self.perplexity
    }

    pub fn max_iter(&self) -> usize {
        self.max_iter
    }

    pub fn preliminary_iter(&self) -> &Option<usize> {
        &self.preliminary_iter
    }

    pub fn rng(&self) -> &R {
        &self.rng
    }
}

#[derive(Debug, Clone, PartialOrd, PartialEq)]
pub struct TSneParams<F, R>(TSneValidParams<F, R>);

impl<F: Float> TSneParams<F, SmallRng> {
    /// Create a t-SNE param set with given embedding size
    ///
    /// # Defaults to:
    ///  * `approx_threshold`: 0.5
    ///  * `perplexity`: 5.0
    ///  * `max_iter`: 2000
    ///  * `rng`: SmallRng with seed 42
    pub fn embedding_size(embedding_size: usize) -> TSneParams<F, SmallRng> {
        Self::embedding_size_with_rng(embedding_size, SmallRng::seed_from_u64(42))
    }
}

impl<F: Float, R: Rng + Clone> TSneParams<F, R> {
    /// Create a t-SNE param set with given embedding size and random number generator
    ///
    /// # Defaults to:
    ///  * `approx_threshold`: 0.5
    ///  * `perplexity`: 5.0
    ///  * `max_iter`: 2000
    pub fn embedding_size_with_rng(embedding_size: usize, rng: R) -> TSneParams<F, R> {
        Self(TSneValidParams {
            embedding_size,
            rng,
            approx_threshold: F::cast(0.5),
            perplexity: F::cast(5.0),
            max_iter: 2000,
            preliminary_iter: None,
        })
    }

    /// Set the approximation threshold of the Barnes Hut algorithm
    ///
    /// The threshold decides whether a cluster centroid can be used as a summary for the whole
    /// area. This was proposed by Barnes and Hut and compares the ratio of cell radius and
    /// distance to a factor theta. This threshold lies in range (0, inf) where a value of 0
    /// disables approximation and a positive value approximates the gradient with the cell center.
    pub fn approx_threshold(mut self, threshold: F) -> Self {
        self.0.approx_threshold = threshold;

        self
    }

    /// Set the perplexity of the t-SNE algorithm
    pub fn perplexity(mut self, perplexity: F) -> Self {
        self.0.perplexity = perplexity;

        self
    }

    /// Set the maximal number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.0.max_iter = max_iter;

        self
    }

    /// Set the number of iterations after which the true P distribution is used
    ///
    /// At the beginning of the training process it is useful to multiply the P distribution values
    /// by a certain factor (here 12x) to get the global view right. After this number of iterations
    /// the true P distribution value is used. If None the number is estimated.
    pub fn preliminary_iter(mut self, num_iter: usize) -> Self {
        self.0.preliminary_iter = Some(num_iter);

        self
    }
}

impl<F: Float, R> ParamGuard for TSneParams<F, R> {
    type Checked = TSneValidParams<F, R>;
    type Error = TSneError;

    /// Validates parameters
    fn check_ref(&self) -> Result<&Self::Checked, Self::Error> {
        if self.0.perplexity.is_negative() {
            Err(TSneError::NegativePerplexity)
        } else if self.0.approx_threshold.is_negative() {
            Err(TSneError::NegativeApproximationThreshold)
        } else {
            Ok(&self.0)
        }
    }

    fn check(self) -> Result<Self::Checked, Self::Error> {
        self.check_ref()?;
        Ok(self.0)
    }
}
