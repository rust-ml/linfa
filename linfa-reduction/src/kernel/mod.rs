pub mod dot;
pub mod gaussian;
pub mod sparse_gaussian;
pub use dot::DotKernel;
pub use gaussian::GaussianKernel;
pub use sparse_gaussian::SparseGaussianKernel;

use std::iter::Sum;

use ndarray::prelude::*;
use ndarray::Data;
use sprs::CsMat;

use crate::Float;
use crate::diffusion_map::{DiffusionMapHyperParams, DiffusionMap};
use crate::pca::PrincipalComponentAnalysis;

/// Symmetric kernel function, can be sparse or dense
///
/// Required functions are:
///  * `mul_normalized_similarity`: performs a matrix multiplication with `rhs`, which is a
///  normalized (sum of columns and rows equal one) symmetric similarity matrix
///  * `size`: returns the number of data-points
pub trait Kernel<A: Float> {
    fn mul_similarity(&self, rhs: &ArrayView2<A>) -> Array2<A>;
    fn sum(&self) -> Array1<A>;
    fn size(&self) -> usize;

}

impl<S: Data<Elem = A>, A: Float> Kernel<A> for ArrayBase<S, Ix2> {
    fn mul_similarity(&self, rhs: &ArrayView2<A>) -> Array2<A> {
        self.dot(rhs)
    }

    fn sum(&self) -> Array1<A> {
        self.sum_axis(Axis(1))
    }

    fn size(&self) -> usize {
        assert_eq!(self.ncols(), self.nrows());

        self.ncols()
    }
}

impl<A: Float> Kernel<A> for CsMat<A> {
    fn mul_similarity(&self, rhs: &ArrayView2<A>) -> Array2<A> {
        self * rhs
    }

    fn sum(&self) -> Array1<A> {
        let mut sum = Array1::zeros(self.cols());

        for (val, i) in self.iter() {
            let (_, col) = i;
            sum[col] += *val;
        }

        sum
    }

    fn size(&self) -> usize {
        assert_eq!(self.cols(), self.rows());

        self.cols()
    }
}

/// Converts an object into a kernel
pub trait IntoKernel<A: Float> {
    type IntoKer: Kernel<A>;

    fn into_kernel(self) -> Self::IntoKer;

    fn reduce_fixed(self, embedding_size: usize) -> DiffusionMap<A>
    where Self: Sized
    {
        let params = DiffusionMapHyperParams::new(embedding_size)
            .steps(1)
            .build();

        DiffusionMap::project(params, self)
    }
}

pub enum Method {
    Gaussian { eps: f32 },
}

#[cfg(test)]
mod tests {
    use super::Kernel;
    use sprs::CsMatBase;
    use ndarray::{Array1, Array2};
    use ndarray_rand::{rand_distr::Uniform, RandomExt};

    #[test]
    fn test_sprs() {
        let a: Array2<f64> = Array2::random((10, 10), Uniform::new(0., 1.));
        let a = CsMatBase::csr_from_dense(a.view(), 1e-5);

        assert_eq!(a.size(), 10);
        assert_eq!(a.apply_gram(Array2::eye(10)), a.to_dense());
    }

    #[test]
    fn test_dense() {
        let id: Array2<f64> = Array2::eye(10);

        assert_eq!(Kernel::sum(&id), Array1::ones(10));
        assert_eq!(Kernel::sum(&id), Array1::ones(10));

        assert_eq!(id.size(), 10);
    }
}
