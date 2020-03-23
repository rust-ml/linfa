use ndarray::prelude::*;
use ndarray::Data;
use ndarray::NdFloat;
use num_traits::{FromPrimitive, Num};
use sprs::CsMat;

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
