use linfa::Float;
use ndarray::{Array, Array2};
use ndarray_rand::{
    rand_distr::{Normal, StandardNormal},
    RandomExt,
};
use rand::{
    distributions::{Bernoulli, Distribution, Standard},
    Rng,
};
use sprs::{CsMat, TriMat};

use crate::ReductionError;

pub trait ProjectionMethod {
    type RandomDistribution;
    type ProjectionMatrix<F: Float>
    where
        Self::RandomDistribution: Distribution<F>;

    fn generate_matrix<F: Float>(
        n_features: usize,
        n_dims: usize,
        rng: &mut impl Rng,
    ) -> Result<Self::ProjectionMatrix<F>, ReductionError>
    where
        Self::RandomDistribution: Distribution<F>;
}

pub struct Gaussian;

impl ProjectionMethod for Gaussian {
    type RandomDistribution = StandardNormal;
    type ProjectionMatrix<F: Float>
        = Array2<F>
    where
        StandardNormal: Distribution<F>;

    fn generate_matrix<F: Float>(
        n_features: usize,
        n_dims: usize,
        rng: &mut impl Rng,
    ) -> Result<Self::ProjectionMatrix<F>, ReductionError>
    where
        StandardNormal: Distribution<F>,
    {
        let std_dev = F::cast(n_features).sqrt().recip();
        let gaussian = Normal::new(F::zero(), std_dev)?;

        Ok(Array::random_using((n_features, n_dims), gaussian, rng))
    }
}

pub struct Sparse;

impl ProjectionMethod for Sparse {
    type RandomDistribution = Standard;
    type ProjectionMatrix<F: Float>
        = CsMat<F>
    where
        Standard: Distribution<F>;

    fn generate_matrix<F: Float>(
        n_features: usize,
        n_dims: usize,
        rng: &mut impl Rng,
    ) -> Result<Self::ProjectionMatrix<F>, ReductionError>
    where
        Standard: Distribution<F>,
    {
        let scale = (n_features as f64).sqrt();
        let p = 1f64 / scale;
        let dist = SparseDistribution::new(F::cast(scale), p);

        let (mut row_inds, mut col_inds, mut values) = (Vec::new(), Vec::new(), Vec::new());
        for row in 0..n_features {
            for col in 0..n_dims {
                if let Some(x) = dist.sample(rng) {
                    row_inds.push(row);
                    col_inds.push(col);
                    values.push(x);
                }
            }
        }

        // `proj` will be used as the RHS of a matrix multiplication in [`SparseRandomProjection::transform`],
        // therefore we convert it to `csc` storage.
        let proj = TriMat::from_triplets((n_features, n_dims), row_inds, col_inds, values).to_csc();

        Ok(proj)
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
