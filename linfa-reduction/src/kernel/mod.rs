pub mod gaussian;
pub use gaussian::GaussianKernel;

use ndarray::prelude::*;
use ndarray::Data;
use ndarray::NdFloat;
use num_traits::{FromPrimitive, Num};
use sprs::CsMat;

use crate::Reduced;
use crate::{DiffusionMap, DiffusionMapHyperParams, PrincipalComponentAnalysis};

/// Symmetric kernel function, can be sparse or dense
///
/// Required functions are:
///  * `apply_gram`: performs a matrix multiplication with `rhs`
///  * `sum`: returns the sum of columns or rows
///  * `size`: returns the number of data-points
pub trait Kernel<A> {
    fn apply_gram<S: Data<Elem = A>>(&self, rhs: ArrayBase<S, Ix2>) -> Array2<A>;
    fn sum(&self) -> Array1<A>;
    fn size(&self) -> usize;
}

impl<S: Data<Elem = A>, A: NdFloat + 'static + FromPrimitive> Kernel<A> for ArrayBase<S, Ix2> {
    fn apply_gram<T: Data<Elem = A>>(&self, rhs: ArrayBase<T, Ix2>) -> Array2<A> {
        self.dot(&rhs)
    }

    fn sum(&self) -> Array1<A> {
        self.sum_axis(ndarray::Axis(0))
    }

    fn size(&self) -> usize {
        assert_eq!(self.ncols(), self.nrows());

        self.ncols()
    }
}

impl<A: NdFloat + 'static + FromPrimitive + Num + Default> Kernel<A> for CsMat<A> {
    fn apply_gram<S: Data<Elem = A>>(&self, rhs: ArrayBase<S, Ix2>) -> Array2<A> {
        self * &rhs
    }

    fn sum(&self) -> Array1<A> {
        let mut mean = Array1::zeros(self.cols());

        for (val, (_, col)) in self.iter() {
            mean[col] += *val;
        }

        mean
    }

    fn size(&self) -> usize {
        assert_eq!(self.cols(), self.rows());

        self.cols()
    }
}

/// Converts an object into a kernel
pub trait IntoKernel<A> {
    type IntoKer: Kernel<A>;

    fn into_kernel(self) -> Self::IntoKer;
}

pub trait KernelWithMethod {
    type IntoKer: Kernel<f64>;

    fn kernel_with(&self, method: Method) -> Self::IntoKer;
}

impl KernelWithMethod for Array2<f64> {
    type IntoKer = Kernels;

    fn kernel_with(&self, method: Method) -> Self::IntoKer {
        match method {
            Method::Gaussian { eps } => Kernels::Gaussian(GaussianKernel::new(&self, eps)),
        }
    }
}


pub enum Method {
    Gaussian { eps: f64 },
}

pub enum Kernels {
    Gaussian(GaussianKernel)
}

impl Kernels {
    pub fn reduce(self, embedding_size: usize, method: crate::Method) -> impl Reduced {
        match method {
            crate::Method::DiffusionMap { steps } => {
                let params = DiffusionMapHyperParams::new(embedding_size)
                    .steps(steps)
                    .build();

                DiffusionMap::project(params, self)
            },
            crate::Method::PrincipalComponentAnalysis => {
                PrincipalComponentAnalysis::fit(self, embedding_size)
            }
        }
    }
}

impl Kernel<f64> for Kernels {
    fn apply_gram<T: Data<Elem = f64>>(&self, rhs: ArrayBase<T, Ix2>) -> Array2<f64> {
        match self {
            Kernels::Gaussian(x) => x.data.apply_gram(rhs),
        }
    }

    fn sum(&self) -> Array1<f64> {
        match self {
            Kernels::Gaussian(x) => Kernel::sum(&x.data),
        }
    }

    fn size(&self) -> usize {
        match self {
            Kernels::Gaussian(x) => x.data.size(),
        }
    }
}

impl IntoKernel<f64> for Kernels {
    type IntoKer = Kernels;

    fn into_kernel(self) -> Self::IntoKer {
        self
    }
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
